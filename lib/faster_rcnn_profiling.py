from helper import GPURuntimeProfiler
from lib.faster_rcnn_efficient import FasterRCNN

profiler = GPURuntimeProfiler()

class ProfilingFasterRCNN(FasterRCNN):
    """
    This class is like a wrapper around the FasterRCNN base class.
    Every method is decorated with a profiling method that measures the gpu time
    that a method uses.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @profiler.measure_gpu('forward_backbone')
    def forward_backbone(self, *args, **kwargs):
        return super().forward_backbone(*args, **kwargs)

    @profiler.measure_gpu('forward_rpn')
    def forward_rpn(self, *args, **kwargs):
        return super().forward_rpn(*args, **kwargs)

    @profiler.measure_gpu('forward_nms')
    def forward_nms(self, *args, **kwargs):
        return super().forward_nms(*args, **kwargs)

    @profiler.measure_gpu('forward_roi_pooling')
    def forward_roi_pooling(self, *args, **kwargs):
        return super().forward_roi_pooling(*args, **kwargs)

    @profiler.measure_gpu('forward_rcnn')
    def forward_rcnn(self, *args, **kwargs):
        return super().forward_rcnn(*args, **kwargs)

    @profiler.measure_gpu('forward')
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
