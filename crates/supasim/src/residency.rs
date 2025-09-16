use std::collections::HashMap;

use parking_lot::{Condvar, Mutex};

use crate::{BufferRange, InstanceState, SupaSimResult};

pub struct BufferResidency {}

struct SliceTrackerInner {
    uses: HashMap<BufferUserId, BufferUser>,
    current_id: u64,
    cpu_locked: Option<u64>,
}
pub struct SliceTracker {
    condvar: Condvar,
    /// Contains a submission index.
    ///
    /// Also, the higher level u64 is for the current id. Bool is for whether cpu work is prevented(used for mapping logic)
    mutex: Mutex<SliceTrackerInner>,
}
impl SliceTracker {
    pub fn new() -> Self {
        Self {
            condvar: Condvar::new(),
            mutex: Mutex::new(SliceTrackerInner {
                uses: HashMap::new(),
                current_id: 0,
                cpu_locked: None,
            }),
        }
    }
    pub fn acquire<B: hal::Backend>(
        &self,
        instance: &InstanceState<B>,
        range: BufferRange,
        user: BufferUser,
        bypass_gpu: bool,
    ) -> SupaSimResult<B, BufferUserId> {
        let mut lock = self.mutex.lock();
        if !bypass_gpu {
            let mut cont = true;
            let mut gpu_submissions = Vec::new();
            while cont {
                let mut has_cpu = false;
                cont = false;
                gpu_submissions.clear();
                if user.submission_id().is_none() && lock.cpu_locked.is_some() {
                    cont = true;
                    gpu_submissions.push(lock.cpu_locked.unwrap());
                }
                for (&a, &submission) in &lock.uses {
                    if a.range.overlaps(&range) {
                        // If this is part of the same GPU submission, don't try to wait
                        if submission.submission_id() == user.submission_id()
                            && submission.submission_id().is_some()
                        {
                            continue;
                        }
                        cont = true;
                        if let Some(sub) = submission.submission_id() {
                            gpu_submissions.push(sub);
                        } else {
                            has_cpu = true;
                        }
                        break;
                    }
                }
                if cont && has_cpu {
                    self.condvar.wait(&mut lock);
                } else if cont {
                    let sub = *gpu_submissions.iter().min().unwrap();
                    if sub == u64::MAX {
                        self.condvar.wait(&mut lock);
                    } else {
                        drop(lock);
                        instance.sync_thread().wait_for(sub, true)?;
                        lock = self.mutex.lock();
                    }
                } else {
                    break;
                }
            }
        }
        let id = BufferUserId {
            range,
            id: lock.current_id,
        };
        lock.current_id += 1;
        lock.uses.insert(id, user);
        Ok(id)
    }
    pub fn update_user_submission<B: hal::Backend>(
        &self,
        user: BufferUserId,
        submission_id: u64,
        instance: &InstanceState<B>,
    ) {
        let mut lock = self.mutex.lock();
        if instance
            .sync_thread()
            .wait_for(submission_id, false)
            .unwrap()
        {
            lock.uses.remove(&user).unwrap();
        } else {
            lock.uses
                .get_mut(&user)
                .unwrap()
                .set_submission_id(submission_id);
        }
        self.condvar.notify_all();
    }
    pub fn release(&self, range: BufferUserId) {
        self.mutex.lock().uses.remove(&range);
        self.condvar.notify_all();
    }
    pub fn acquire_cpu<B: hal::Backend>(&self, submission_id: u64) -> SupaSimResult<B, ()> {
        let mut lock = self.mutex.lock();
        let mut cont = true;
        while cont {
            cont = false;
            for &submission in lock.uses.values() {
                if submission.submission_id().is_none() {
                    cont = true;
                    break;
                }
            }
            if cont {
                self.condvar.wait(&mut lock);
            }
        }
        lock.cpu_locked = Some(submission_id);
        Ok(())
    }
    pub fn release_cpu(&self) {
        self.mutex.lock().cpu_locked = None;
        self.condvar.notify_all();
    }
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct BufferUserId {
    pub range: BufferRange,
    pub id: u64,
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum BufferUser {
    Gpu(u64),
    Cpu,
    /// CPU submission in a queue
    Cross(u64),
}
impl BufferUser {
    fn submission_id(&self) -> Option<u64> {
        match self {
            Self::Gpu(a) | Self::Cross(a) => Some(*a),
            Self::Cpu => None,
        }
    }
    fn set_submission_id(&mut self, id: u64) {
        match self {
            Self::Gpu(v) => *v = id,
            Self::Cross(v) => *v = id,
            Self::Cpu => panic!(),
        }
    }
}
