// SPDX-License-Identifier: Apache-2.0
#include "resume.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using namespace joint_tail;
using Words=std::array<uint32_t,4>;
using Bytes=std::array<uint8_t,656>;
static void check(bool v,const char*m){if(!v)throw std::runtime_error(m);}
static Bytes body(uint32_t id){
    Bytes b{};const uint8_t c[]={0xc4,0xc0,0xb8,0,0x38,0x40,0x44};
    for(unsigned i=0;i<512;++i)b[i]=c[((id>>((i/8)%18))+i)%7];
    for(unsigned i=0;i<64;++i){uint16_t h=uint16_t(0x3e80|(((id>>((i/4)%18))+i)&127));memcpy(b.data()+512+2*i,&h,2);}
    const float v[]={.03125f,.0625f,.125f,.25f};memcpy(b.data()+640,v,sizeof(v));return b;
}
struct Port{
    Saved s{};uint64_t key{},head=2046,cursor=15,base=2046;uint32_t arrivals=0;
    uint32_t ring[16]{},ends[4]{},n=0,count=0;std::vector<Words> list;
    Port(){for(uint64_t i=15;i<31;++i)ring[i%16]=uint32_t(i/16&1U)<<2;}
    uint32_t Depth(){return 16;} uint64_t Key(){return key;} uint64_t Producer(){return head;}
    uint64_t Cursor(){return cursor;} uint32_t Read(uint32_t i){return ring[i];}
    uint64_t Timeout(){return 31;} uint64_t Invalid(){return 32;} uint64_t Failure(){return 33;}uint32_t Limit(){return 4;}
    void Issue(uint64_t next){check(next>head&&next==base+ends[s.issued],"wrong batch WQE boundary");head=next;}
    void Consume(uint64_t next,bool ok){check(ok&&next==cursor+1,"wrong acknowledgement");cursor=next;}
};
struct Row{Port q[4];int parent=-1;bool done=false,prefetch=false,started=false;};
int main(int argc,char**argv){try{
    check(argc==4,"input/events/oracle required");std::ifstream in(argv[1]),ev(argv[2]),oracle(argv[3],std::ios::binary);
    unsigned B;in>>B;check(B==32,"expected BS32");std::vector<Row> rows(B);
    std::map<std::pair<unsigned,unsigned>,Bytes> out;unsigned posted=0,seen=0,parts=0;
    for(unsigned r=0;r<B;++r)for(unsigned q=0;q<4;++q){
        auto&p=rows[r].q[q];unsigned a,b;in>>a>>b>>p.n>>p.count;check(a==r&&b==q&&p.count<=4,"fixture grouping");
        for(unsigned j=0;j<p.count;++j)in>>p.ends[j];
        p.key=100+r*4+q;
        for(unsigned j=0;j<p.n;++j){Words w;for(auto&v:w)in>>v;check(w[3]==r+1,"foreign destination row");p.list.push_back(w);}
    }
    double t;std::string op;unsigned r,q,k;
    while(ev>>t>>op>>r>>q>>k){
        auto&row=rows.at(r);
        if(op=="start"){check(!row.started,"double row preparation");row.started=true;
            for(unsigned j=0;j<4;++j){auto&p=row.q[j];check(Begin(p.s,r,j)&&!Begin(p.s,r,j),"begin twice");
                check(Arm(p.s,p.head,p.cursor,p.count,p.ends,p.n,false,2,p.Key()),"bad native batch");}
        }else if(op=="park"){
            auto&dst=rows.at(q);check(!dst.started && dst.parent<0,"second-claim");dst.parent=r;++parts;
            for(auto&p:row.q)check(Peek(p.s,p)==Probe::Empty&&p.s.seen==0,"visible CQ at park");
        }else if(op=="retire"){
            check(row.prefetch&&!row.done,"missing Q2 or doubled retirement");
            for(auto&p:row.q)check(StartRetirement(p.s)&&!StartRetirement(p.s)&&!p.s.error,"unsafe row retirement");
            row.done=true;
        }else if(op=="index"){check(q<4&&row.started,"bad Index event");
        }else if(op=="prefetch"){check(!row.prefetch&&q==1,"duplicate or wrong Q2");row.prefetch=true;
        }else{auto&p=row.q[q];
            if(op=="post"){if(row.parent>=0)for(auto&parent:rows[row.parent].q)
                    check(Gate(parent.s),"child posted early");
                check(k==p.s.issued&&PostOne(p.s,p),"duplicate/out-of-order issue");++posted;
            }else if(op=="ack"){check(k==p.arrivals&&k<p.s.issued,"orphan CQE");
                const auto at=15+p.arrivals++;p.ring[at%16]=uint32_t(1^((at/16)&1))<<2;
                for(unsigned j=k?p.ends[k-1]:0;j<p.ends[k];++j){auto w=p.list[j];
                    check(out.emplace(std::make_pair(w[3],w[1]),body(w[2])).second,"duplicate destination slot");}
            }else if(op=="cq"){const auto old=p.cursor;
                check(k==p.s.seen && Peek(p.s,p)==Probe::Ready && Peek(p.s,p)==Probe::Ready && old==p.cursor,"unexpected CQE");
                PollOne(p.s,p);check(p.cursor==old+1&&!p.s.error,"native CQ did not advance");++seen;
            }else check(false,"unknown event");
        }
    }
    for(auto&row:rows){check(row.done,"unretired row");for(auto&p:row.q)for(auto&w:p.list){
        Words ref;Bytes bytes;oracle.read(reinterpret_cast<char*>(ref.data()),sizeof(ref));
        oracle.read(reinterpret_cast<char*>(bytes.data()),bytes.size());check(bool(oracle)&&w==ref,"address/list order mismatch");
        check(out.at({w[3],w[1]})==bytes,"KV byte mismatch");}}
    check(oracle.peek()==EOF&&seen==posted,"extra oracle records or CQ outstanding");
    std::cout<<"PASS rows="<<B<<" q="<<B*4<<" parked="<<parts<<" post/cq="<<posted<<"/"<<seen
             <<" exact_ordered_WQE="<<out.size()<<" payload_bytes="<<out.size()*656<<"\n";
    return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}
