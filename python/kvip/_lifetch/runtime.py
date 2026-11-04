"""Host tiling parameters; fixed-size ABI, no per-forward allocation or copy."""

import ctypes


class RuntimeShape(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in [
            "batch",
            "history",
            "resident",
            "indexHeads",
            "aicCount",
            "splitQ",
            "liCores",
            "rounds",
            "serviceAics",
            "serviceLanes",
            "finalWorkers",
            "finalCores",
            "finalPostOffset",
            "finalOwnerBytes",
            "tokenBytes",
            "postGroups",
            "postThreads",
            "postBank",
            "batchedRows",
            "scanTokens",
        ]
    ]

    @classmethod
    def prepare(
        cls,
        library,
        *,
        batch,
        history,
        resident,
        index_heads,
        aic_count,
        windows,
        split=True,
        service_aics=0,
        final_workers=0,
        token_bytes=656,
    ):
        shape = cls(
            batch=batch,
            history=history,
            resident=resident,
            indexHeads=index_heads,
            aicCount=aic_count,
            splitQ=int(split),
            rounds=windows,
            serviceAics=service_aics,
            finalWorkers=final_workers,
            tokenBytes=token_bytes,
        )
        function = library.kvip_prepare_shape
        function.argtypes = [ctypes.POINTER(cls)]
        function.restype = ctypes.c_uint32
        if function(ctypes.byref(shape)):
            raise ValueError("Unsupported LIFetch runtime shape or core/queue budget")
        return shape
