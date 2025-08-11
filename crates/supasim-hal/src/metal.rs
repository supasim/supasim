use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLCommandQueue;
use thiserror::Error;

use crate::{Backend, BackendInstance};

#[derive(Debug, Clone)]
pub struct Metal;
impl Metal {
    pub fn create_instance() -> Result<MetalInstance, MetalError> {
        todo!()
    }
}
impl Backend for Metal {
    type Instance = MetalInstance;
    type Error = MetalError;
}

#[derive(Debug)]
pub struct MetalInstance {
    command_queue: ProtocolObject<dyn MTLCommandQueue>,
}
impl BackendInstance<Metal> for MetalInstance {}

#[derive(Error, Debug)]
pub enum MetalError {}
impl crate::Error<Metal> for MetalError {
    fn is_out_of_device_memory(&self) -> bool {
        false
    }
    fn is_out_of_host_memory(&self) -> bool {
        false
    }
    fn is_timeout(&self) -> bool {
        false
    }
}
