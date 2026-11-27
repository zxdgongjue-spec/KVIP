import struct
import unittest
from lease_guard import check_lease,disjoint

class Lease(unittest.TestCase):
    def setUp(self):
        self.e=bytearray(160*192);self.args=bytearray(160*112)
        for i in range(160):
            o=i*192;b=0x100000+i*0x10000
            struct.pack_into("<QQ",self.e,o+8,0xb0000000,0x1000000)
            struct.pack_into("<QQ",self.e,o+32,0xa0000000,0x1000000)
            struct.pack_into("<QQ",self.e,o+56,b,b+0x5000)
            struct.pack_into("<I",self.e,o+76,256);struct.pack_into("<I",self.e,o+84,64)
            struct.pack_into("<QQ",self.e,o+104,b+0x6000,b+0x7000)
            struct.pack_into("<II",self.e,o+124,64,32)
            struct.pack_into("<10Q8I",self.args,112*i,0,0,0,0,0,0,0xa0000000,b+0x8000,b+0x9000,b+0xa000,0,0,0,656,0,0,0,0)
    def check(self):return check_lease(self.e,self.args,2048,1216)
    def test_valid_shared_read_only_remote(self):self.assertEqual(len(self.check()[0]),960)
    def test_remote_aliases_output(self):
        struct.pack_into("<Q",self.e,8,0xa0000800)
        with self.assertRaises(ValueError):self.check()
    def test_sq_cross_type_overlap(self):
        struct.pack_into("<Q",self.e,192+104,0x100000+8)
        with self.assertRaises(ValueError):self.check()
    def test_partially_overlapping_cq(self):
        struct.pack_into("<Q",self.e,192+104,0x106000+32)
        with self.assertRaises(ValueError):self.check()
    def test_shared_db(self):
        struct.pack_into("<Q",self.e,192+112,0x107000)
        with self.assertRaises(ValueError):self.check()
    def test_status_overlaps_queue(self):
        struct.pack_into("<Q",self.args,112+56,0x10a000-1152)
        with self.assertRaises(ValueError):self.check()
    def test_hbm_overlap_with_index_source(self):
        with self.assertRaises(ValueError):disjoint([(0xa0000000,1024,"KV out"),(0xa0000300,1024,"Index key source")])
    def test_partially_shared_destination(self):
        with self.assertRaises(ValueError):disjoint([(2048,2048,"hot destination"),(4095,2048,"next hot destination")])
if __name__=="__main__":unittest.main()
