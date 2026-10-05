"""Captured execution of the existing frozen CLIP vision; same feature contract.

Persistent static buffers are serialized. Capture parity is a software gate,
not a physical invariance theorem. No encoder training or precision change.
"""

from threading import Lock
from time import perf_counter

from .frozen_clip import FrozenClip,ClipFeatures,prepare224,shifted_unit_feature


class CapturedFrozenClip(FrozenClip):
    def __init__(self,checkpoint,*,device='cuda:0'):
        super().__init__(checkpoint,device=device)
        torch=self.torch;self._mutex=Lock()
        with torch.inference_mode(),torch.jit.optimized_execution(False):
            self._input=torch.zeros((1,3,224,224),device=self.device,dtype=torch.float16)
            stream=torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(5):self.model(self._input)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self._graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):self._output=self.model(self._input)
            cases=[torch.zeros_like(self._input),
                   torch.linspace(-1,1,3*224*224,device=self.device).reshape_as(self._input).half()]
            generator=torch.Generator(device=self.device).manual_seed(20261006)
            cases.append(torch.randn(self._input.shape,device=self.device,dtype=torch.float16,generator=generator))
            captured=[]
            for value in cases:
                expected=self.model(value)
                self._input.copy_(value);self._graph.replay()
                actual=self._output.clone()
                if not torch.equal(expected,actual) or not torch.isfinite(actual).all():
                    raise ValueError('Capture changed synthetic feature values')
                captured.append(actual)
            # A stale buffer would emit the zero-input result for another input.
            stale_rejected=not torch.equal(captured[0],captured[1])
            self._input.copy_(cases[1]);self._graph.replay()
            if not stale_rejected or not torch.equal(captured[1],self._output):
                raise ValueError('Capture input refresh/repeat control failed')
            torch.cuda.synchronize(self.device)
        self.provenance.update({'execution':'serialized fixed-shape CUDA graph replay',
                                'synthetic_capture_parity_exact':True,'capture_control_cases':3,
                                'stale_buffer_negative_rejected':True,'capture_repeat_exact':True,
                                'capture_warmup_side_stream':True})

    def extract(self,image):
        start=perf_counter();pixels=prepare224(image);ready=perf_counter();torch=self.torch
        with self._mutex,torch.inference_mode(),torch.jit.optimized_execution(False):
            value=torch.from_numpy(pixels.copy()).unsqueeze(0).to(self.device).half()
            self._input.copy_(value);self._graph.replay()
            values=self._output[0].float().cpu().numpy()
            torch.cuda.synchronize(self.device)
        return ClipFeatures(shifted_unit_feature(values),1000*(ready-start),1000*(perf_counter()-ready))
