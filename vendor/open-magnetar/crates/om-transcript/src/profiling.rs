//! Optional wall-clock stage profiling for the isolated RAM benchmark.
//!
//! Enable the `profiling` feature on `om-transcript` to collect timings. Spans
//! record exclusive time: a parent's duration excludes its nested spans. A span
//! around a Rayon call measures the caller's wall time, including its wait for
//! workers. Collection is thread-local; create and finish a span on one thread,
//! and call [`clear`] / [`take`] on that thread between measured operations.
//!
//! Without the feature, span construction/destruction and collection are no-ops.

#[cfg(feature = "profiling")]
mod enabled {
    use std::{
        cell::RefCell,
        marker::PhantomData,
        rc::Rc,
        time::{Duration, Instant},
    };

    #[derive(Default)]
    struct Collector {
        active: Vec<(u64, Duration)>,
        totals: Vec<(&'static str, Duration)>,
        next_id: u64,
    }

    thread_local! {
        static COLLECTOR: RefCell<Collector> = RefCell::new(Collector::default());
    }

    /// A stage whose exclusive wall-clock duration is recorded when dropped.
    ///
    /// Finish nested spans before their enclosing span. The marker prevents
    /// moving a live span to another thread.
    pub struct Span {
        name: &'static str,
        started: Instant,
        id: u64,
        _thread: PhantomData<Rc<()>>,
    }

    impl Span {
        #[inline]
        pub fn new(name: &'static str) -> Self {
            let started = Instant::now();
            let id = COLLECTOR.with(|collector| {
                let mut collector = collector.borrow_mut();
                let id = collector.next_id;
                collector.next_id = collector.next_id.wrapping_add(1);
                collector.active.push((id, Duration::ZERO));
                id
            });
            Self {
                name,
                started,
                id,
                _thread: PhantomData,
            }
        }
    }

    impl Drop for Span {
        fn drop(&mut self) {
            let elapsed = self.started.elapsed();
            COLLECTOR.with(|collector| {
                let mut collector = collector.borrow_mut();
                let (id, children) = collector.active.pop().expect("profiling span stack");
                assert_eq!(id, self.id, "profiling spans must finish in nesting order");
                if let Some((_, parent_children)) = collector.active.last_mut() {
                    *parent_children += elapsed;
                }
                let exclusive = elapsed.saturating_sub(children);
                if let Some((_, total)) = collector
                    .totals
                    .iter_mut()
                    .find(|(name, _)| *name == self.name)
                {
                    *total += exclusive;
                } else {
                    collector.totals.push((self.name, exclusive));
                }
            });
        }
    }

    /// Discard completed measurements on the calling thread.
    pub fn clear() {
        COLLECTOR.with(|collector| collector.borrow_mut().totals.clear());
    }

    /// Return and clear completed measurements, summed by stage name.
    pub fn take() -> Vec<(&'static str, Duration)> {
        COLLECTOR.with(|collector| std::mem::take(&mut collector.borrow_mut().totals))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn nested_stages_do_not_double_count_and_take_clears() {
            clear();
            let wall = Instant::now();
            {
                let _parent = Span::new("parent");
                {
                    let _child = Span::new("child");
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
            let wall = wall.elapsed();
            let timings = take();
            assert_eq!(timings.len(), 2);
            assert!(timings.iter().any(|(name, _)| *name == "parent"));
            assert!(timings.iter().any(|(name, _)| *name == "child"));
            assert!(timings.iter().map(|(_, time)| *time).sum::<Duration>() <= wall);
            assert!(take().is_empty());
        }

        #[test]
        fn repeated_stages_aggregate_and_threads_are_independent() {
            clear();
            drop(Span::new("repeat"));
            drop(Span::new("repeat"));
            let worker = std::thread::spawn(|| {
                assert!(take().is_empty());
                drop(Span::new("worker"));
                take()
            })
            .join()
            .unwrap();
            assert_eq!(worker.len(), 1);
            assert_eq!(worker[0].0, "worker");
            let timings = take();
            assert_eq!(timings.len(), 1);
            assert_eq!(timings[0].0, "repeat");
        }
    }
}

#[cfg(feature = "profiling")]
pub use enabled::{Span, clear, take};

#[cfg(not(feature = "profiling"))]
mod disabled {
    use std::time::Duration;

    /// A no-op stage marker when profiling is disabled.
    pub struct Span;

    impl Span {
        #[inline(always)]
        pub fn new(_name: &'static str) -> Self {
            Self
        }
    }

    #[inline(always)]
    pub fn clear() {}

    #[inline(always)]
    pub fn take() -> Vec<(&'static str, Duration)> {
        Vec::new()
    }
}

#[cfg(not(feature = "profiling"))]
pub use disabled::{Span, clear, take};
