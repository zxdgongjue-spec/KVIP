// SPDX-License-Identifier: Apache-2.0
#include "resume.h"
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
using namespace joint_tail;
struct Port {
    uint64_t cursor,upper;uint32_t depth;uint32_t attempts=0;
    std::array<uint32_t,16> ring{};std::vector<uint64_t> bells,commits;
    uint64_t Key(){return reinterpret_cast<uint64_t>(this);} uint64_t Producer(){return upper;}
    uint32_t Depth(){return depth;}uint64_t Cursor(){return cursor;}
    uint32_t Limit(){return 4;}uint64_t Timeout(){return 113;}
    uint64_t Invalid(){return 114;}uint64_t Failure(){return 115;}
    uint32_t Read(uint32_t slot){++attempts;return ring.at(slot);}
    void Consume(uint64_t next,bool){assert(next==cursor+1);cursor=next;commits.push_back(next);}
    void Issue(uint64_t head){assert(head>upper);upper=head;bells.push_back(head);}
    void write(uint64_t seq,int kind){
        ring.at(seq%depth)=(uint32_t(((seq/depth)&1)^(kind!=0))<<2)|(kind==2?1U<<16:0);
    }
};
Saved prepared(Port&p,bool first=true) {
    Saved s{};const uint32_t ends[]{1,5,8,13};
    assert(Begin(s,17,2));assert(!Begin(s,17,2));
    assert(Arm(s,100,p.cursor,4,ends,13,first,2,p.Key()));assert(!Arm(s,100,p.cursor,4,ends,13,first,2,p.Key()));
    p.upper=s.sqHead;if(first)p.Issue(101);
    for(unsigned i=0;i<p.depth;++i)p.write(p.cursor+i,0);
    return s;
}
int main(){
  unsigned cases=0;
  for(unsigned d:{3u,4u,5u,8u,16u})for(uint64_t h=0;h<3*d;++h)
   for(int error=-1;error<4;++error){
    Port p{h,0,d};auto s=prepared(p);auto duplicate=s;
    assert(!Begin(s,4,0));assert(std::memcmp(&s,&duplicate,sizeof(s))==0);
    unsigned expected=0;
    for(unsigned j=0;j<4;++j){
      PostOne(s,p);PostOne(s,p);
      if(s.seen==s.issued)break;
      auto head=p.cursor;auto n=p.commits.size();p.write(head,0);
      assert(Peek(s,p)==Probe::Empty&&Peek(s,p)==Probe::Empty);
      assert(p.cursor==head&&p.commits.size()==n);
      p.write(head,error==int(j)?2:1);
      assert(Peek(s,p)==(error==int(j)?Probe::Error:Probe::Ready));assert(p.commits.size()==n);
      PollOne(s,p);++expected;
      if(error==int(j))assert(!PostOne(s,p)&&s.error==115);
    }
    assert(s.seen==expected && Settled(s)&&StartRetirement(s)&&!StartRetirement(s));
    assert(!Begin(s,4,0)&&p.commits.size()==p.bells.size());
    assert(s.success==expected-(error>=0));
    ++cases;
   }
  Port p{7,0,8};auto s=prepared(p);PostOne(s,p);
  for(unsigned i=0;i<4;++i)PollOne(s,p);
  assert(s.stop && s.error==113 && s.issued-s.seen==2 && p.commits.empty());
  Port a{3,0,4},b{3,0,4};auto x=prepared(a),y=prepared(b,false);
  PostOne(x,a);assert(!Gate(x)&&b.bells.empty());
  for(unsigned i=0;i<3;i++){a.write(a.cursor,1);PollOne(x,a);PostOne(x,a);}
  assert(Gate(x)&&!Begin(y,17,2)&&PostOne(y,b)&&b.bells.size()==1&&y.issued==1);
  assert(a.commits.size()==3&&b.commits.empty());b.cursor++;PollOne(y,b);
  assert(y.stop && y.error==114 && b.commits.empty());
  Port foreign{7,0,8},owner{7,0,8};auto k=prepared(owner);
  assert(Peek(k,foreign)==Probe::Stale && foreign.attempts==0 && !PostOne(k,foreign) && foreign.bells.empty());
  auto mismatch=prepared(foreign);++foreign.upper;assert(!PostOne(mismatch,foreign) && mismatch.stop);
  Port pre{3,0,4};for(unsigned i=0;i<3;++i)pre.write(3+i,0);
  pre.write(5,1);assert(!PrefetchVisible(pre,3,4,3));
  pre.write(3,1);assert(!PrefetchVisible(pre,3,4,3));
  pre.write(4,1);assert(PrefetchVisible(pre,3,4,3)&&pre.commits.empty());
  pre.write(5,0);pre.write(4,2);assert(PrefetchVisible(pre,3,4,3)&&pre.commits.empty());
  assert(!PrefetchVisible(pre,3,4,4));
  const uint32_t bad[]={1,4,4,8};Saved inv{};assert(Begin(inv,12,1));
  assert(!Arm(inv,3,2,4,bad,8,false,2,99));
  assert(Disjoint({1,3},{4,1})&&!Disjoint({1,4},{4,1})&&!Disjoint({UINT64_MAX-1,2},{1,1}));
  std::cout<<"PASS "<<cases<<" wrapped FIFO/error sequences, held-next-row, no-consume peek, stale cursor, timeout, duplicate prepare/retirement and overflow"<<std::endl;
}
