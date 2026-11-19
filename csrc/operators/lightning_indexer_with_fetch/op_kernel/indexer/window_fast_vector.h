#pragma once
// Native LI bitmap layout; preserve exact candidates and the existing TopK primitive.
namespace lifetch_incremental {
__simd_vf__ inline void MarkNonemptyTiles(__ubuf__ uint16_t* bitmap,__ubuf__ uint16_t* flags,uint32_t n){
 using namespace MicroAPI;auto all=CreateMask<uint32_t,MaskPattern::ALL>();RegTensor<int32_t> seq;RegTensor<uint32_t> offsets,part,bits,value;MaskReg nonempty,valid;
 Arange(seq,int32_t(0));Muls(offsets,reinterpret_cast<RegTensor<uint32_t>&>(seq),uint32_t(8),all);
 for(uint16_t t=0;t<128;t+=64){
  Duplicate(bits,uint32_t(0));
  Adds(value,reinterpret_cast<RegTensor<uint32_t>&>(seq),uint32_t(t),all);
  CompareScalar<uint32_t,CMPMODE::LT>(valid,value,n/128,all);
  for(uint16_t j=0;j<8;++j){Gather(part,reinterpret_cast<__ubuf__ uint32_t*>(bitmap)+t*8+j,offsets,valid);Or(bits,bits,part,all);}
  Duplicate(value,uint32_t(0x55555555));And(bits,bits,value,all);CompareScalar<uint32_t,CMPMODE::NE>(nonempty,bits,uint32_t(0),all);
  MaskAnd(nonempty,nonempty,valid,all);
  Duplicate<uint32_t,MaskMergeMode::ZEROING>(value,uint32_t(128),nonempty);
  StoreAlign<uint32_t,StoreDist::DIST_PACK_B32>(reinterpret_cast<__ubuf__ uint32_t*>(flags+t),value,all);
 }
}
__simd_vf__ inline void PackTileList(__ubuf__ uint16_t* flags,__ubuf__ uint32_t* counts){
 using namespace MicroAPI;auto all=CreateMask<uint16_t,MaskPattern::ALL>();auto one=CreateMask<uint32_t,MaskPattern::VL1>();
 RegTensor<uint16_t> f,ones,packed;RegTensor<int16_t> ids;RegTensor<uint32_t> total;MaskReg selected;UnalignRegForStore store;
 LoadAlign(f,flags);ReduceSum(total,f,all);StoreAlign(counts,total,one);
 CompareScalar<uint16_t,CMPMODE::GT>(selected,f,uint16_t(0),all);
 Duplicate<uint16_t,MaskMergeMode::ZEROING>(ones,uint16_t(1),selected);ReduceSum(total,ones,all);StoreAlign(counts+8,total,one);
 Arange(ids,int16_t(0));ClearSpr<SpecialPurposeReg::AR>();
 Squeeze<uint16_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,reinterpret_cast<RegTensor<uint16_t>&>(ids),selected);
 StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(flags,packed,store);StoreUnAlignPost(flags,store);
}
__simd_vf__ inline void CompactSparseTiles(__ubuf__ uint16_t* raw,__ubuf__ uint16_t* bitmap,__ubuf__ uint16_t* tiles,
 __ubuf__ uint16_t* packed,__ubuf__ uint16_t* mapping,__ubuf__ uint32_t* counts,uint32_t blocks,uint16_t base,uint16_t cutoff){
 using namespace MicroAPI;auto all=CreateMask<uint16_t,MaskPattern::ALL>();auto one=CreateMask<uint32_t,MaskPattern::VL1>();
 RegTensor<uint16_t> values,sv,si,ones,sum;RegTensor<int16_t> ids;RegTensor<uint32_t> total;
 MaskReg selected,above;UnalignRegForStore scoreStore,idStore;Duplicate(sum,uint16_t(0));ClearSpr<SpecialPurposeReg::AR>();
 for(uint16_t k=0;k<blocks;++k){uint16_t i=tiles[k]*128;
  LoadAlign(selected,bitmap+i/8);LoadAlign(values,raw+i);CompareScalar<uint16_t,CMPMODE::GT>(above,values,cutoff,all);MaskAnd(selected,selected,above,all);
  Duplicate<uint16_t,MaskMergeMode::ZEROING>(ones,uint16_t(1),selected);Add(sum,sum,ones,all);Arange(ids,int16_t(base+i));
  Squeeze<uint16_t,MicroAPI::GatherMaskMode::STORE_REG>(sv,values,selected);StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(packed,sv,scoreStore);
  StoreAlign(bitmap+i/8,selected);
 }
 StoreUnAlignPost(packed,scoreStore);ReduceSum(total,sum,all);StoreAlign(counts,total,one);
 ClearSpr<SpecialPurposeReg::AR>();
 for(uint16_t k=0;k<blocks;++k){uint16_t i=tiles[k]*128;LoadAlign(selected,bitmap+i/8);Arange(ids,int16_t(base+i));
  Squeeze<uint16_t,MicroAPI::GatherMaskMode::STORE_REG>(si,reinterpret_cast<RegTensor<uint16_t>&>(ids),selected);StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(mapping,si,idStore);}
 StoreUnAlignPost(mapping,idStore);
}
}
namespace lifetch_incremental {
__simd_vf__ inline void BuildAddedMap(__ubuf__ uint16_t* added,__ubuf__ uint32_t* table,uint32_t n,uint32_t base){
 using namespace MicroAPI;auto all=CreateMask<uint32_t,MaskPattern::ALL>();RegTensor<uint16_t> v;RegTensor<uint32_t> out;
 for(uint16_t i=0;i<n;i+=64){LoadAlign<uint16_t,LoadDist::DIST_UNPACK_B16>(v,added+i);Adds(out,reinterpret_cast<RegTensor<uint32_t>&>(v),base,all);StoreAlign(table+i,out,all);}
}
__simd_vf__ inline void MapAndExport(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* table,__ubuf__ uint16_t* scores,__ubuf__ uint32_t* cold,__ubuf__ uint32_t* stats,uint32_t resident,uint32_t valid,bool exportCold=true){
 using namespace MicroAPI;auto all=CreateMask<uint32_t,MaskPattern::ALL>();auto one=CreateMask<uint32_t,MaskPattern::VL1>();
 RegTensor<uint32_t> ids,mapped,minimum,values,total,packed,ones,sum;RegTensor<uint16_t> score;MaskReg isCold,bounded,take;UnalignRegForStore store;
 Duplicate(minimum,uint32_t(65535));Duplicate(sum,uint32_t(0));ClearSpr<SpecialPurposeReg::AR>();
 for(uint16_t i=0;i<2048;i+=64){LoadAlign(ids,selected+i);Gather(mapped,table,ids,all);StoreAlign(selected+i,mapped,all);
  LoadAlign<uint16_t,LoadDist::DIST_UNPACK_B16>(score,scores+i);Min(minimum,minimum,reinterpret_cast<RegTensor<uint32_t>&>(score),all);
  if(exportCold) {
  CompareScalar<uint32_t,CMPMODE::GE>(isCold,mapped,resident,all);CompareScalar<uint32_t,CMPMODE::LT>(bounded,mapped,valid,all);MaskAnd(take,isCold,bounded,all);
  Duplicate<uint32_t,MaskMergeMode::ZEROING>(ones,uint32_t(1),take);Add(sum,sum,ones,all);
  Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,mapped,take);StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(cold,packed,store);
  }
 }
 if(exportCold)StoreUnAlignPost(cold,store);ReduceMin(total,minimum,all);StoreAlign(stats,total,one);ReduceSum(total,sum,all);StoreAlign(stats+8,total,one);
}
}
