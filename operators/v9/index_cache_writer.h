#pragma once
namespace rotating {
// Binding: key, scale, new key, new scale, logical write slots, block tables,
// owner pointers, logical lengths, candidate lengths, columns, C, S, batch.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void WriteOwnedIndexQ4(__gm__ uint64_t* args, uint32_t b) {
  uint32_t tid=threadIdx.x;
  if(b>=args[12] || tid>=128) return;
  // Optional graph guards; eager supplies null pointers.
  if(args[13] && asc_ldcg(reinterpret_cast<__gm__ int32_t*>(args[13]))==2) return;
  if(args[14] && asc_ldcg(reinterpret_cast<__gm__ int32_t*>(args[14])+b)!=2) return;
  uint32_t q=tid/32, word=tid%32, row=b*4+q;
  int32_t token=reinterpret_cast<__gm__ int32_t*>(args[7])[b]-4+int32_t(q);
  int32_t length=reinterpret_cast<__gm__ int32_t*>(args[8])[b];
  uint64_t base=reinterpret_cast<__gm__ uint64_t*>(args[6])[b];
  uint32_t capacity=uint32_t(args[10]), hotCapacity=uint32_t(args[11]);
  int32_t position=-1;
  if(base && token>=0 && token<length) {
    auto positions=reinterpret_cast<__gm__ int32_t*>(base+128+uint64_t(hotCapacity)*132+
        uint64_t(hotCapacity+capacity)*8);
    position=asc_ldcg(positions+token);
  }
  uint64_t key,scale;
  if(position<=-2) {
    uint32_t hot=uint32_t(-position-2);
    key=base+128+uint64_t(hot)*128;
    scale=base+128+uint64_t(hotCapacity)*128+uint64_t(hot)*4;
  } else {
    int64_t physical;
    if(base && token>=0 && token<length && position>=0) {
      auto table=reinterpret_cast<__gm__ int32_t*>(args[5]);
      physical=table[b*args[9]+uint32_t(position)/128]*128+position%128;
    } else physical=reinterpret_cast<__gm__ int64_t*>(args[4])[row];
    if(physical<0) return; // PAD slots never address HBM.
    key=args[0]+uint64_t(physical)*128;
    scale=args[1]+uint64_t(physical)*4;
  }
  asc_stcg(reinterpret_cast<__gm__ uint32_t*>(key)+word,
           asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(args[2])+row*32+word));
  if(!word) asc_stcg(reinterpret_cast<__gm__ uint32_t*>(scale),
                    asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(args[3])+row));
  asc_threadfence();
}
} // namespace rotating
