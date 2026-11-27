"""Pre-capture lease check for the isolated, fixed-shape BS32 trial."""
import ctypes
import struct
import weakref
from math import ceil, isfinite

def disjoint(spans):
    prev_end=0
    for start,size,name in sorted(spans):
        if not start or not size or start<prev_end or start+size >= 1<<64:
            raise ValueError(f"Invalid or aliased terminal span {name}")
        prev_end=start+size

def check_lease(entities,args,required_rows,status_bytes):
    if status_bytes < 64 or required_rows<1:
        raise ValueError("Missing queue or row bounds")
    if len(entities)!=160*192 or len(args)!=160*112:
        raise ValueError("Expected one current lease with 160 live views")
    extents=[]; common=None; sources=set()
    for i in range(160):
        arg=struct.unpack_from("<10Q8I",args,112*i)
        e=entities[192*i:192*(i+1)]
        sq,sqd=struct.unpack_from("<QQ",e,56)
        depth=struct.unpack_from("<I",e,76)[0]
        stride=struct.unpack_from("<I",e,84)[0]
        cq,cqd=struct.unpack_from("<QQ",e,104)
        cdepth,cstride=struct.unpack_from("<II",e,124)
        remote,remote_size=struct.unpack_from("<QQ",e,8)
        sources.add((remote,remote_size))
        output,capacity=struct.unpack_from("<QQ",e,32)
        if (arg[6]!=output or capacity < required_rows*arg[13] or not arg[13]
            or (common is not None and common!=(output,capacity,arg[13]))
            or depth<2 or depth&(depth-1) or stride!=64 or cdepth<=2 or cstride<4):
            raise ValueError(f"Unusable queue geometry on live view {i}")
        common=(output,capacity,arg[13])
        extents.extend((base,size,f"{i}/{kind}") for base,size,kind in (
            (sq,depth*stride,"SQ"),(sqd,4,"SQ doorbell"),(cq,cdepth*cstride,"CQ"),
            (cqd,4,"CQ doorbell"),(arg[9],64,"cursor"),(arg[7],status_bytes,"status")))
    for remote,remote_size in sources:
        # Conservative numerical check. Repeated read-only source ranges are legal.
        disjoint(extents+[(common[0],common[1],"registered HBM"),(remote,remote_size,"remote source")])

    return extents, common, sources

def span(t,name):
    if not t.is_contiguous():
        raise ValueError(f"Expected contiguous fixed-batch buffer {name}")
    return t.data_ptr(),t.numel()*t.element_size(),name

def checked_workspace(fwd, counter_mhz=None):
    """Call once outside graph capture. Workspace and lease must live through graph completion."""
    rot=fwd.rotating;term=fwd.terminal;torch=rot.torch
    if (fwd.batch!=32 or term is None or term.rotating is not rot or
        rot.runtime_shape.finalWorkers!=4 or len(term.queue_views)!=128):
        raise ValueError("Expected fixed BS32 Q4 direct lease")
    stream=rot.owner.lease.check_stream();stream.synchronize()
    raw=rot.owner.device_args[:160].cpu().contiguous().numpy().tobytes()
    if any(struct.unpack_from("<Q",raw,112*i+64)[0] != term.entity_base+i*192 for i in range(160)):
        raise ValueError("Current lease has unexpected entity addresses")
    copied=term.args.cpu().contiguous().numpy().tobytes()
    if copied!=b"".join(raw[112*i:112*(i+1)] for i in term.queue_views):
        raise ValueError("Terminal binding belongs to a different lease")
    e=torch.empty(160*192,device=rot.device,dtype=torch.uint8)
    snapshot=rot.owner.lease.library.kvip_transport_snapshot
    if snapshot(term.entity_base,e.data_ptr(),e.numel(),stream.npu_stream):
        raise ValueError("Live entity snapshot failed")
    u=fwd.update.bindings.cpu().tolist()
    stride=rot.output_stride if hasattr(rot,"output_stride") else None
    if not isinstance(stride,int) or stride <= 0:
        raise ValueError("Missing verified per-owner output stride")
    output_rows=struct.unpack_from("<I",raw,88)[0]
    if output_rows not in (32*stride,33*stride) or any(
        struct.unpack_from("<I",raw,112*i+88)[0]!=output_rows for i in range(160)):
        raise ValueError("Fixed BS32 requires exactly 32 or 33 registered owner rows")
    ports,hbm,sources=check_lease(e.cpu().numpy().tobytes(),raw,output_rows,rot.owner.abi_sizes["status"])
    keys,scales,addresses,logical=fwd.update.storage[:4]
    owner_count=output_rows//stride
    if any(t.shape[0]!=owner_count for t in (keys,scales,addresses,logical,rot.hot_valid)):
        raise ValueError("Owner allocation differs between Index, resident bitmap and KV")
    # Partial aliases crossing row boundaries invalidate parking. The two documented exact
    # output aliases are safe only for the identity routes rechecked on each device invocation.
    buffers=[span(v,k) for k,v in (
        ("keys",keys),("scales",scales),("addresses",addresses),("logical map",logical),
        ("owner",fwd.update.storage[4]),("changes",rot.changes),("slots",rot.change_slots),
        ("mapping",rot.mapping),("reserved hot",rot.reserved_hot),("incremental",rot.window_topk_storage),("hot valid",rot.hot_valid),("locations",rot.locations),
        ("selected",rot.arguments[9]),("rows",rot.rows),("journal",rot.mapping_journal),
        ("retirement journal",rot.owners),("private tail status",rot.final_status),("event memory",rot.events),("LRU",rot.lru_slots),("count",rot.change_count),("status",rot.row_status),
        ("output flags",term.flags),("trace",term.times),("capacity",term.capacity),
        ("diagnostic sources",fwd.events_src),("diagnostic slots",fwd.events_dst),("query errors",fwd.query_errors),("Index scratch",fwd.dense_shadow)) if v is not None]
    if u[0:4]!=[x[0] for x in buffers[:4]] or u[19]!=rot.locations.data_ptr():
        raise ValueError("Index binding disagrees with actual buffers")
    for name,value,base in (("public IDs",fwd.update.logical_output,rot.arguments[9]),
                            ("public slots",fwd.update.slot_output,rot.locations)):
        if value.data_ptr()!=base.data_ptr():buffers.append(span(value,name))
    work=torch.zeros(128*(1+32*26),device=rot.device,dtype=torch.uint8)
    disjoint(ports+buffers+[span(work,"scheduling mailbox"),(hbm[0],hbm[1],"registered HBM")])
    for remote,size in sources:
        disjoint(buffers+[(remote,size,"remote source"),span(work,"scheduling mailbox")])
    hz=fwd.calibrate_counter_mhz() if counter_mhz is None else counter_mhz
    if not isfinite(hz) or hz<=0:raise ValueError("Invalid device counter calibration")
    return work,ceil(32*hz)

class BoundJointTerminal:
    """Explicit test entry for a caller that has already issued LI for this same binding."""
    def __init__(self, forward, library, counter_mhz=None):
        self.fwd=weakref.proxy(forward)
        self.workspace,self.guard_ticks=checked_workspace(forward, counter_mhz)
        self.library=library
        self.function=library.kvip_test_two_rows
        self.function.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint64]
        self.function.restype=ctypes.c_uint32
        self.pipeline=library.kvip_test_forward_two_rows
        self.pipeline.argtypes=[ctypes.c_void_p]*7+[ctypes.c_uint64]
        self.pipeline.restype=ctypes.c_uint32
        self.library_path=library._name

    def forward(self, launch, hot, bindings, terminal, stream, shape):
        """Record the original hot/LI cold launches, followed by this binding's private two-row tail."""
        return self.pipeline(bindings,terminal,stream,shape,launch,hot,self.workspace.data_ptr(),self.guard_ticks)

    def readback(self):
        """Read after graph completion and the full output checks, once per individual bound state."""
        words=self.workspace.view(self.fwd.torch.int64).reshape(-1,16)[:,0].cpu().tolist()
        if words[0]!=1:
            raise ValueError(f"Two-row kernel did not accept the live owner/output route (mode={words[0]})")
        pairs=[]
        for row in range(32):
            box=words[1+row*26:1+(row+1)*26]
            child=box[24]
            if 1<=child<=32:
                if child==row+1 or any(v!=1 for v in box[4:24]):
                    raise ValueError(f"Incomplete Q1–Q4 handoff from {row} to {child-1}: {box}")
                pairs.append((row,child-1))
        if len({r for pair in pairs for r in pair})!=2*len(pairs):
            raise ValueError("A terminal row was prepared twice")
        return dict(accepted=True,parked_rows=len(pairs),ordered_pairs=pairs,
                    guard_ticks=self.guard_ticks)


    def launch(self):
        stream=self.fwd.rotating.owner.lease.check_stream()
        result=self.function(64,stream.npu_stream,self.fwd.terminal.workspace.data_ptr(),
                             self.workspace.data_ptr(),self.guard_ticks)
        if result:
            raise RuntimeError(f"Experimental terminal launch failed ({result})")
