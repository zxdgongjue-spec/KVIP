"""Exercise capture-time B24 metadata with fake addresses; no device submission."""
import struct
import unittest
from types import SimpleNamespace as NS
from lease_guard import checked_workspace, BoundJointTerminal

class Tensor:
    def __init__(self,addr,shape=(1,),size=4,blob=None):
        self.addr=addr;self.shape=shape;self.size=size;self.blob=blob
    def data_ptr(self):return self.addr
    def numel(self):
        n=1
        for v in self.shape:n*=v
        return n
    def element_size(self):return self.size
    def is_contiguous(self):return True
    def cpu(self):return self
    def contiguous(self):return self
    def numpy(self):return self
    def tobytes(self):return bytes(self.blob)
    def tolist(self):return list(self.blob)
    def __getitem__(self,key):return self
class Fixture:
    def __init__(self):
        self.address=0x11000000000
        def tensor(shape=(1,),size=4,blob=None):
            self.address+=0x80000000
            return Tensor(self.address,shape,size,blob)
        self.array=tensor
        def snap(src,dst,num,stream):
            self.temp.blob=entities
            return int(src!=entity or num!=160*192 or dst!=self.temp.data_ptr() or stream!=322)
        def alloc(n,**_):
            self.temp=tensor((n,),1);return self.temp
        torch=NS(empty=alloc,zeros=alloc,uint8=1)
        r=NS(torch=torch,device="npu:0",runtime_shape=NS(finalWorkers=4),output_stride=16896)
        u=NS();self.f=NS(batch=32,q_workers=2,rotating=r,update=u)
        names="changes change_slots mapping reserved_hot window_topk_storage locations rows mapping_journal owners final_status events lru_slots change_count row_status".split()
        for n in names:setattr(r,n,tensor())
        for n in "events_src events_dst query_errors dense_shadow".split():setattr(self.f,n,tensor())
        r.hot_valid=tensor((33,16384))
        r.arguments=[None]*9+[tensor()]
        u.storage=tuple(tensor((33,131072)) for _ in range(4))+(tensor(),)
        u.logical_output=tensor();u.slot_output=r.locations
        ptr=[0]*28;ptr[:4]=[v.data_ptr() for v in u.storage[:4]];ptr[19]=r.locations.data_ptr()
        u.bindings=Tensor(2,blob=ptr)
        entity=0x80000000;entities=bytearray(160*192);raw=bytearray(160*112)
        for i in range(160):
            e=i*192;base=0x300000+i*0x10000
            struct.pack_into("<QQ",entities,e+8,0x900000000,0x100000)
            struct.pack_into("<QQ",entities,e+32,0x100000000,33*16896*656)
            struct.pack_into("<QQ",entities,e+56,base,base+0x5000)
            struct.pack_into("<II",entities,e+76,256,0);struct.pack_into("<I",entities,e+84,64)
            struct.pack_into("<QQ",entities,e+104,base+0x6000,base+0x7000)
            struct.pack_into("<II",entities,e+124,64,32)
            struct.pack_into("<10Q8I",raw,i*112,0,0,0,0,0,0,0x100000000,base+0x8000,entity+192*i,base+0xa000,0,8192,33*16896,656,1,1,1,0)
        views=[2*i+1 if j==0 else 64+3*i+j-1 for i in range(32) for j in range(4)]
        cropped=b"".join(raw[112*i:112*i+112] for i in views)
        t=NS(rotating=r,args=Tensor(4,blob=cropped),entity_base=entity,queue_views=views)
        for n in "flags times capacity".split():setattr(t,n,tensor())
        self.f.terminal=t
        r.owner=NS(lease=NS(library=NS(kvip_transport_snapshot=snap),check_stream=lambda:NS(npu_stream=322,synchronize=lambda:None)),
                   device_args=Tensor(3,blob=raw),abi_sizes={"status":1216})

class Verify(unittest.TestCase):
    def test_b24_standard_sentinel(self):
        f=Fixture()
        work,ticks=checked_workspace(f.f,49.6)
        self.assertEqual((work.numel(),ticks),(106624,1588))
        f.f.rotating.hot_valid.shape=(32,16384)
        with self.assertRaisesRegex(ValueError,"Owner allocation"):checked_workspace(f.f,49.6)
    def test_final_geometry(self):
        f=Fixture();f.f.rotating.runtime_shape.finalWorkers=2
        with self.assertRaisesRegex(ValueError,"BS32 Q4"):checked_workspace(f.f,49.6)
    def test_readback(self):
        class WS(list):
            def view(self,*a):return self
            def reshape(self,*a):return self
            def __getitem__(self,k):return self if isinstance(k,tuple) else super().__getitem__(k)
            def cpu(self):return self
            def tolist(self):return list(self)
        words=WS([0]*833)
        bound=NS(workspace=words,fwd=NS(torch=NS(int64=1)),guard_ticks=1632)
        with self.assertRaisesRegex(ValueError,"did not accept"):BoundJointTerminal.readback(bound)
        words[0]=1
        self.assertEqual(BoundJointTerminal.readback(bound)["parked_rows"],0)
        v=1+7*26
        words[v+4:v+24]=[1]*20;words[v+24]=30
        self.assertEqual(BoundJointTerminal.readback(bound)["ordered_pairs"],[(7,29)])
        words[v+13]=2
        with self.assertRaisesRegex(ValueError,"Incomplete"):BoundJointTerminal.readback(bound)
if __name__=="__main__":unittest.main()
