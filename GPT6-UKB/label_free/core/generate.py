#!/usr/bin/env python3
"""Dictionary-only synthetic baseline measurements; all priors are assumptions."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from scipy.special import expit, ndtr, ndtri

VERSION = "dictionary-only-physiology-v1"
EXPECTED_SCHEMA_SHA256 = "a9b18cf55c3464b49aee5abe7f9fd84b7430e04ae7c7ca534af3a07c40f2effd"
MODEL_REQUEST = {"model_requested":"gpt-6-astra","reasoning_effort_requested":"ultra",
                 "backend_verified":False,"note":"Requested identity and effort; hidden backend not verified."}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_schema(path):
    if sha(path) != EXPECTED_SCHEMA_SHA256:
        raise ValueError("Schema differs from the permitted frozen dictionary")
    with open(path,newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=201 or len({r["Field_ID"] for r in rows})!=201:
        raise ValueError("Expected 201 distinct measurement IDs")
    return rows

class Population:
    def __init__(self,schema,n,seed):
        self.schema,self.n,self.seed=schema,n,seed
        self.rows={r["Field_ID"]:r for r in schema}
        self.x,self.rules,self.missing={},{},{}
    def u(self,name):
        key=hashlib.sha256(f"{VERSION}|{self.seed}|{name}".encode()).digest()
        return np.random.Generator(np.random.PCG64(int.from_bytes(key[:16],"little"))).random(self.n)
    def z(self,name):
        return ndtri(np.clip(self.u(name),1e-12,1-1e-12))
    def bern(self,name,p):
        return (self.u(name)<p).astype(float)
    def tn(self,name,mean,sd,lo,hi):
        lower,upper=ndtr((lo-mean)/sd),ndtr((hi-mean)/sd)
        return mean+sd*ndtri(np.clip(lower+(upper-lower)*self.u(name),1e-12,1-1e-12))
    def ordinal(self,name,score,codes):
        thresholds=np.linspace(-1.5,1.5,len(codes)-1)
        rank=np.sum((score+.7*self.z(name))[:,None]>thresholds,axis=1)
        return np.asarray(codes)[rank].astype(float)
    def put(self,fid,value,rule):
        fid=str(fid)
        if fid not in self.rows or fid in self.x:
            raise ValueError(f"Unknown or duplicate field {fid}")
        self.x[fid]=np.broadcast_to(np.asarray(value,dtype=float),(self.n,)).copy()
        self.rules[fid]=rule
    def mask(self,fid,condition,reason):
        fid=str(fid)
        self.x[fid][condition]=np.nan
        self.missing.setdefault(fid,[]).append(reason)
    def generate(self,apply_missingness=True):
        z,u,B,O,P,C,S=self.z,self.u,self.bern,self.ordinal,self.put,np.clip,expit
        # Each named Z is independent N(0,1); shared quantities explicitly induce dependence.
        sex=B("sex",.48); age=self.tn("age",55,8,39,70); a=(age-55)/10
        access,drive=z("access"),z("drive")
        activity=.35*access-.20*a+.90*z("activity")
        diet=.35*access+.25*activity+.85*z("diet")
        adip=-.30*activity-.18*diet+.18*a+.90*z("adiposity")
        hydration=z("hydration"); inflammation=.25*adip+.90*z("inflammation")
        muscle=.30*activity+.85*z("muscle"); bone=z("bone"); pigment=z("pigmentation")
        P(21022,age,"TN(55,8;39,70); a=(age−55)/10")
        P(31,sex,"Bernoulli(0.48); assumed 0=female, 1=male")
        education=C(np.rint(18+2.5*access-.35*a+1.4*z("education")),12,30)
        employed=B("employed",S(3.1-.13*(age-45)+.20*access))
        workhours=employed*C(36+4*sex+7*z("workhours"),8,65)
        commute=employed*C(np.exp(1.8+.30*access+.70*z("commute")),.2,120)
        P(845,education,"clip(round(18+2.5 access−0.35 a+1.4 Z),12,30)")
        P(757,employed*C(10+3.5*a+3*z("jobyears"),0,age-education),"employed × clip(10+3.5 a+3 Z,0,age−education)")
        P(767,workhours,"employed × clip(36+4 sex+7 Z,8,65)")
        P(796,commute,"employed × clip(exp(1.8+0.30 access+0.70 Z),0.2,120)")
        P(777,employed*C(np.rint(workhours/8+.5*z("commutedays")),1,6),"employed × clip(round(workhours/8+0.5 Z),1,6)")
        P(699,C(12+5*a+6*z("address"),0,age),"clip(12+5 a+6 Z,0,age)")
        household=1+B("cohabit",S(1.2-.25*a))+B("child1",S(-.6-.9*a))+B("child2",S(-1.5-.9*a))
        P(709,household,"1+Bernoulli(σ(1.2−0.25 a))+Bernoulli(σ(−0.6−0.9 a))+Bernoulli(σ(−1.5−0.9 a))")

        smoke_score=-.35*access+.25*drive+.20*z("smokepropensity")
        current=B("current_smoke",S(-2+smoke_score-.20*a))
        former=(1-current)*B("former_smoke",S(-.75+.30*a+.30*smoke_score))
        smoking=2*current+former
        duration=C(age-(18+2*z("smokestart"))-former*(8+4*S(z("quittime"))),0,age-12)
        intensity=C(np.exp(2.1+.45*z("smokeintensity")),1,45)
        pack=(current+former)*duration*intensity/20
        P(20116,smoking,"2 current+former; assumed 0=never,1=former,2=current; probabilities in MODEL_SPEC")
        P(1239,current*(1+(intensity<5)),"0 if noncurrent; 1 if current intensity≥5 cigarettes/day; 2 otherwise (assumed coding)")
        P(1249,np.where(smoking==0,4,np.where(intensity>=5,1,2)),"assumed 4=never,1=regular past use,2=occasional past use; code 3 unused")
        P(20161,pack,"(current+former) × smoking_duration × cigarettes_per_day / 20")
        other=B("othersmoker",S(-2+.9*current+.2*(household-1)))
        P(1259,np.minimum(2,other+current),"min(2,other_smoker+current); other_smoker∼Bernoulli(σ(−2+0.9 current+0.2(household−1))); exposure rank assumed")
        P(1269,C((other+current)*np.exp(1.5+.5*z("homesmoke")),0,70),"clip((other_smoker+current) exp(1.5+0.5 Z),0,70)")
        P(1279,C(np.exp(.5+.4*current-.25*access+.7*z("outsidesmoke"))*B("outsideexposure",.55),0,60),"clip(exp(0.5+0.4 current−0.25 access+0.7 Z) Bernoulli(0.55),0,60)")

        md=C(np.rint(3+1.25*activity+z("moddays")),0,7)
        vd=C(np.rint(1.8+1.2*activity-.25*a+z("vigdays")),0,7)
        wd=C(np.rint(5+activity+z("walkdays")),0,7)
        for fid,value,rule in [(884,md,"clip(round(3+1.25 activity+Z),0,7)"),(904,vd,"clip(round(1.8+1.2 activity−0.25 a+Z),0,7)"),(864,wd,"clip(round(5+activity+Z),0,7)")]: P(fid,value,rule)
        for fid,days,median,coef,label in [(894,md,35,.20,"moderate"),(914,vd,25,.25,"vigorous"),(874,wd,45,.20,"walking")]:
            P(fid,np.where(days>0,C(median*np.exp(coef*activity+.45*z(label+"duration")),10,180),0),f"0 if corresponding days=0; otherwise clip({median} exp({coef} activity+0.45 Z),10,180)")
        for did,fid,label,offset in [(2634,2624,"heavydiy",-.6),(1021,1011,"lightdiy",.1),(3647,3637,"otherexercise",-.1),(981,971,"pleasurewalk",.5)]:
            propensity=.65*activity+.20*(1-employed)+offset+.5*z(label)
            P(fid,O(label+"freq",propensity,[1,2,3,4,5,6]),f"O(0.65 activity+0.20(1−employed)+{offset}+0.5 Z;1,…,6), increasing frequency assumed")
            P(did,O(label+"duration",.6*propensity,[1,2,3,4,5,6,7]),f"O(0.6[0.65 activity+0.20(1−employed)+{offset}+0.5 Z];1,…,7), increasing duration assumed")
        P(943,O("stairs",.6*activity-.2*a,list(range(6))),"O(0.6 activity−0.2 a;0,…,5), increasing stair use assumed")
        P(924,O("pace",.8*activity-.25*a-.15*adip,[1,2,3]),"O(0.8 activity−0.25 a−0.15 adip;1,2,3), assumed slow/steady/brisk")
        digital=-.55*a+.35*access+.6*z("digital")
        driving=C(np.rint(commute/25*employed+.35*S(drive)),0,4)
        P(1090,driving,"clip(round(commute/25 × employed+0.35 σ(drive)),0,4)")
        P(1080,C(np.rint(2+.8*digital+.3*employed+.7*z("computer")),0,6),"clip(round(2+0.8 digital+0.3 employed+0.7 Z),0,6)")
        P(1070,C(np.rint(2.5+.4*a-.5*activity+.7*z("tv")),0,6),"clip(round(2.5+0.4 a−0.5 activity+0.7 Z),0,6)")
        P(1120,O("phone",digital,list(range(6))),"O(digital;0,…,5), increasing use assumed")
        P(1130,O("handsfree",.5*digital+.3*driving,list(range(5))),"O(0.5 digital+0.3 driving;0,…,4), increasing use assumed")
        P(2237,O("games",.8*digital+.15*sex-.6,[0,1,2]),"O(0.8 digital+0.15 sex−0.6;0,1,2), increasing engagement assumed")
        sleep_quality=.20*activity-.25*drive-.20*adip+.8*z("sleepquality")
        chronotype=-.3*a+.7*z("chronotype")
        P(1160,C(np.rint(7.2+.5*sleep_quality-.15*employed+.7*z("sleephours")),4,10),"clip(round(7.2+0.5 sleep_quality−0.15 employed+0.7 Z),4,10)")
        self.x["1070"]=np.minimum(self.x["1070"],24-self.x["1160"]-self.x["1090"]-self.x["1080"])
        self.rules["1070"]+="; capped at24−sleep−driving−computer hours"
        P(1170,O("waking",chronotype-.5*sleep_quality,[1,2,3,4]),"O(chronotype−0.5 sleep_quality;1,…,4), increasing waking difficulty assumed")
        P(1180,O("chrono",chronotype,[1,2,3,4]),"O(chronotype;1,…,4), morning-to-evening assumed")
        P(1190,O("nap",.35*a-.5*sleep_quality-.6,[1,2,3]),"O(0.35 a−0.5 sleep_quality−0.6;1,2,3), increasing frequency assumed")
        P(1200,O("sleepdisruption",-.8*sleep_quality-.3,[1,2,3]),"O(−0.8 sleep_quality−0.3;1,2,3), increasing frequency assumed")
        P(1210,np.where(B("snore",S(-.5+.65*adip+.5*sex))>0,1,2),"1 with probability σ(−0.5+0.65 adip+0.5 sex), otherwise2; assumed yes/no")
        P(1220,O("doze",-.6*sleep_quality+.2*adip-.8,[0,1,2,3]),"O(−0.6 sleep_quality+0.2 adip−0.8;0,…,3), increasing frequency assumed")

        appetite=.3*sex+.2*activity+.5*z("appetite")
        foods=[(1289,3,.35*diet+.15*appetite,"0.35 diet+0.15 appetite",15,False),(1299,2,.40*diet+.10*appetite,"0.40 diet+0.10 appetite",15,False),(1309,2,.30*diet,"0.30 diet",10,True),(1319,1,.25*diet,"0.25 diet",15,True),(1438,12,.25*appetite-.10*diet,"0.25 appetite−0.10 diet",45,False),(1458,5,.20*diet+.15*appetite,"0.20 diet+0.15 appetite",20,True),(1488,3,.15*a-.10*digital,"0.15 a−0.10 digital",12,False),(1498,2,.2*digital+.15*drive,"0.2 digital+0.15 drive",10,True),(1528,5,.15*activity+.12*diet,"0.15 activity+0.12 diet",15,True)]
        for fid,median,score,expression,maximum,integer in foods:
            value=C(median*np.exp(score+.45*z("food"+str(fid))),0,maximum)
            P(fid,np.rint(value) if integer else value,f"{'round of ' if integer else ''}clip({median} exp({expression}+0.45 Z),0,{maximum})")
        P(1518,O("hotdrink",.15*drive,[1,2,3]),"O(0.15 drive;1,2,3), increasing temperature assumed")
        for fid,score,expression in [(1329,.5*diet-.4,"0.5 diet−0.4"),(1339,.3*diet-.2,"0.3 diet−0.2"),(1349,-.4*diet+.25*appetite-.3,"−0.4 diet+0.25 appetite−0.3"),(1359,.15*diet+.25*appetite,"0.15 diet+0.25 appetite"),(1369,-.2*diet+.3*appetite-.4,"−0.2 diet+0.3 appetite−0.4"),(1379,-.2*diet+.2*appetite-.8,"−0.2 diet+0.2 appetite−0.8"),(1389,-.25*diet+.3*appetite-.5,"−0.25 diet+0.3 appetite−0.5"),(1408,.15*appetite,"0.15 appetite")]:
            P(fid,O("foodfreq"+str(fid),score,list(range(6))),f"O({expression};0,…,5), increasing frequency assumed")
        P(1478,O("salt",-.3*diet+.2*appetite-.3,[1,2,3,4]),"O(−0.3 diet+0.2 appetite−0.3;1,…,4), increasing addition assumed")
        P(6144,np.where(u("exclusion")<S(-2.1+.3*diet),1+np.minimum(3,np.floor(4*u("exclusiontype"))),5),"with probability σ(−2.1+0.3 diet), equal nominal probabilities for assumed 1=eggs,2=dairy,3=wheat,4=sugar; otherwise assumed5=none; single-code representation")
        ac=B("alcoholcurrent",S(1.5+.2*access+.3*drive)); af=(1-ac)*B("alcoholformer",.35)
        alcohol=ac*C(np.exp(1.7+.4*drive+.2*sex+.45*z("alcohol")), .2,45)
        alcohol_score=np.log1p(alcohol)-1.6
        P(20117,2*ac+af,"2 alcohol_current+alcohol_former; current∼Bernoulli(σ(1.5+0.2 access+0.3 drive)); former|noncurrent∼Bernoulli(0.35); assumed0=never,1=former,2=current")
        P(1558,np.select([alcohol==0,alcohol<1,alcohol<3,alcohol<8,alcohol<18],[6,5,4,3,2],default=1),"assumed codes6,5,4,3,2,1 correspond to none,special,monthly,weekly,several/week,daily; amount boundaries0,1,3,8,18 assumed")
        scores=np.stack([.3+.25*access+.4*z("redpref"),.2+.3*(1-sex)+.4*z("whitepref"),.3+.7*sex+.4*z("beerpref"),-.5+.3*drive+.4*z("spiritpref"),-1.5+.2*a+.4*z("fortifiedpref")],axis=1)
        shares=np.exp(scores-scores.max(axis=1,keepdims=True)); shares/=shares.sum(axis=1,keepdims=True)
        for j,fid in enumerate([1568,1578,1588,1598,1608]):
            value=alcohol*shares[:,j]
            P(fid,np.rint(value) if fid==1608 else value,f"{'round of ' if fid==1608 else ''}alcohol_amount × softmax(beverage scores)[{j+1}]; scores defined in MODEL_SPEC; native beverage servings, not ethanol equivalents")
        P(1618,ac*B("alcoholmeal",S(.2+.4*diet+.3*access)),"alcohol_current × Bernoulli(σ(0.2+0.4 diet+0.3 access)); assumed0=no,1=yes")

        summer=C(np.rint(3+.6*activity+.5*(1-employed)+.7*z("summer")),0,8)
        winter=np.minimum(summer,C(np.rint(1.5+.4*activity+.4*(1-employed)+.5*z("winter")),0,6))
        P(1050,summer,"clip(round(3+0.6 activity+0.5(1−employed)+0.7 Z),0,8)")
        P(1060,winter,"min(summer,clip(round(1.5+0.4 activity+0.4(1−employed)+0.5 Z),0,6))")
        skin=O("skin",pigment-.7,[1,2,3,4,5,6])
        P(1717,skin,"O(pigment−0.7;1,…,6), increasing pigmentation assumed; no ancestry inferred")
        P(1727,O("tanning",.75*pigment,[1,2,3,4]),"O(0.75 pigment;1,…,4), increasing tanning response assumed")
        P(1737,C(np.rint(np.exp(1.2-.4*pigment+.15*summer+.5*z("sunburn"))),0,60),"clip(round(exp(1.2−0.4 pigment+0.15 summer+0.5 Z)),0,60)")
        hs=.8*pigment+.65*z("hair")
        P(1747,np.where(u("redhair")<.06*S(-pigment),5,np.select([hs<-.9,hs<-.15,hs<.8],[1,2,3],default=4)),"hair_score=0.8 pigment+0.65 Z; boundaries−0.9,−0.15,0.8 give assumed blond/light brown/dark brown/black codes1..4; code5 red with probability0.06 σ(−pigment);6 unused")
        P(1757,O("facialage",.2*a+.18*summer+.2*current-.4,[1,2,3]),"O(0.2 a+0.18 summer+0.2 current−0.4;1,2,3), younger-to-older appearance assumed")
        P(2267,O("protection",.3*diet-.5*pigment+.15*summer,[1,2,3,4,5]),"O(0.3 diet−0.5 pigment+0.15 summer;1,…,5), increasing protection assumed")
        P(2277,B("solariumany",S(-2.8+.3*drive))*C(np.rint(np.exp(1.6+.6*z("solarium"))),1,60),"Bernoulli(σ(−2.8+0.3 drive)) × clip(round(exp(1.6+0.6 Z)),1,60)")
        P(2139,C(19+.7*access-.8*drive+2*z("debut"),14,age-1),"clip(19+0.7 access−0.8 drive+2 Z,14,age−1)")
        P(2149,C(np.rint(np.exp(1.3+.55*drive+.1*a+.65*z("partners"))),1,150),"clip(round(exp(1.3+0.55 drive+0.1 a+0.65 Z)),1,150)")
        P(2159,B("samesex",.07),"Bernoulli(0.07), independent nominal self-report assumption")
        multiple=B("multiple",.025); maternal=B("maternal",S(-1+.25*a-.3*access))
        P(1777,multiple,"Bernoulli(0.025)")
        P(1787,maternal,"Bernoulli(σ(−1+0.25 a−0.3 access))")
        P(1677,B("breastfed",S(.7+.15*access+.2*a-.3*maternal)),"Bernoulli(σ(0.7+0.15 access+0.2 a−0.3 maternal_smoking))")
        P(20022,C(3.35+.12*sex-.7*multiple-.22*maternal+.45*z("birthweight"),1.2,5.5),"clip(3.35+0.12 sex−0.7 multiple_birth−0.22 maternal_smoking+0.45 Z,1.2,5.5)")
        longevity=.35*access+.8*z("longevity"); deaths=[]
        for lid,did,label,gap,center,lo,hi in [(2946,1807,"father",29,79,10,109),(1845,3526,"mother",27,83,18,104)]:
            pa=age+self.tn(label+"gap",gap,4,18,42)
            life=C(center+5*longevity+8*z(label+"lifetime"),lo,hi); dead=life<=pa; deaths.append(dead)
            P(lid,pa,f"age+TN({gap},4;18,42), observed only for living parent within dictionary range")
            P(did,life,f"clip({center}+5 family_longevity+8 Z,{lo},{hi}), observed only if lifetime≤current parent age")
            row=self.rows[str(lid)]
            self.mask(lid,dead|(pa<float(row["Recommended_Hard_Min"]))|(pa>float(row["Recommended_Hard_Max"])),"Structural: deceased parent or current age outside supplied dictionary support")
            self.mask(did,~dead,"Structural: parent living at baseline")
        P(4501,(deaths[0]|deaths[1]).astype(float)*B("familycontext",.9),"Bernoulli(0.9) × indicator(either modeled parent deceased); baseline family context only")
        burden=.5*a+.4*adip-.4*activity+.8*z("burden")
        P(2188,B("longstanding",S(-1.7+.7*burden)),"Bernoulli(σ(−1.7+0.7 functional_burden)); functional_burden=0.5 a+0.4 adip−0.4 activity+0.8 Z; no diagnosis")

        # Body mass accounting includes an unreported residual head/other compartment.
        height=C(162+13*sex-.05*(age-55)+6.2*z("height"),145,194)
        weight=C((18+25*S(-.75+.55*adip+.16*a))*(height/100)**2,46,160)
        bmi=weight/(height/100)**2
        pf=C(.20+.012*(bmi-23)+.105*(1-sex)+.018*a+.025*z("fatfraction"),
             np.maximum(.09,np.maximum(5.1/weight,1-87.5/weight)),np.minimum(.49,1-36/weight))
        fat,lean=weight*pf,weight*(1-pf)
        P(50,height,"clip(162+13 sex−0.05(age−55)+6.2 Z,145,194)")
        P(21002,weight,"clip([18+25 σ(−0.75+0.55 adip+0.16 a)](height/100)²,46,160)")
        P(21001,bmi,"weight/(height/100)², exact before float32 export")
        P(20015,C(height*(.522+.009*z("sitting")),70,110),"clip(height(0.522+0.009 Z),70,110)")
        P(48,C(79+2.1*(bmi-23)+6*sex+.8*a+3*z("waist"),60,150),"clip(79+2.1(BMI−23)+6 sex+0.8 a+3 Z,60,150)")
        P(49,C(94+1.6*(bmi-23)+3*(1-sex)+3*z("hip"),77,160),"clip(94+1.6(BMI−23)+3(1−sex)+3 Z,77,160)")
        P(23099,100*pf,"100 p; p=clip(0.20+0.012(BMI−23)+0.105(1−sex)+0.018 a+0.025 Z,max(0.09,5.1/weight,1−87.5/weight),min(0.49,1−36/weight))")
        P(23100,fat,"weight × p")
        P(23101,lean,"weight × (1−p); weight=fat+lean")
        water=lean*C(.732+.008*hydration,.71,.75)
        P(23102,water,"lean × clip(0.732+0.008 hydration,0.71,0.75)")
        P(23105,4.184*(370+21.6*lean),"4.184(370+21.6 lean), assumed energy scaling in kJ/day")
        ls=.006*np.tanh(z("legshare")); ars=.004*np.tanh(z("armshare")); asym=.002*np.tanh(z("asymmetry"))
        fl=.01*np.tanh(z("legfatshare")); fa=.005*np.tanh(z("armfatshare"))
        segments=[("left_leg",23115,23116,23117,23118,.185+ls+asym,.18+fl+.5*asym,"0.185+l+s","0.18+f+0.5s"),("right_leg",23111,23112,23113,23114,.185+ls-asym,.18+fl-.5*asym,"0.185+l−s","0.18+f−0.5s"),("left_arm",23123,23124,23125,23126,.06+ars+asym,.045+fa+.5*asym,"0.060+r+s","0.045+g+0.5s"),("right_arm",23119,23120,23121,23122,.06+ars-asym,.045+fa-.5*asym,"0.060+r−s","0.045+g−0.5s"),("trunk",23127,23128,23129,23130,.46+.008*np.tanh(z("trunklean")), .50+.010*np.tanh(z("trunkfat")),"0.46+0.008 tanh(Z)","0.50+0.010 tanh(Z)")]
        segmentlean={}
        for label,pid,fid,lid,mid,lshare,fshare,lexpr,fexpr in segments:
            f,l=fat*fshare,lean*lshare; segmentlean[label]=l
            P(fid,f,f"whole_fat × ({fexpr}); share symbols defined in MODEL_SPEC")
            P(lid,l,f"whole_lean × ({lexpr}); share symbols defined in MODEL_SPEC")
            P(pid,100*f/(f+l),"100 segment_fat/(segment_fat+segment_lean)")
            P(mid,.96*l,"0.96 segment_lean; assumed muscle-like predicted mass excludes4% other lean tissue")
        conductivity=np.exp(.04*z("conductivity"))
        P(23106,C(.8*height**2/water*conductivity,314,973),"clip(0.8 height²/whole_water × conductivity,314,973); conductivity=exp(0.04 Z)")
        for fid,label,coef in [(23110,"left_arm",.032),(23109,"right_arm",.032),(23108,"left_leg",.070),(23107,"right_leg",.070)]:
            row=self.rows[str(fid)]
            P(fid,C(coef*height**2/(segmentlean[label]*.732)*conductivity*np.exp(.025*z(label+"impedance")),float(row["Recommended_Hard_Min"]),float(row["Recommended_Hard_Max"])),f"clip({coef} height²/(segment_lean × 0.732) × conductivity × exp(0.025 Z),dictionary bounds)")
        strength=23+13*sex+.23*(lean-45)+1.8*muscle-1.3*a
        right=C(strength+2*z("rightgrip"),5,70)
        P(47,right,"clip(23+13 sex+0.23(lean−45)+1.8 muscle−1.3 a+2 Z,5,70)")
        P(46,C(right-1+.8*z("gripasym"),4,75),"clip(right_grip−1+0.8 Z,4,75)")
        dbp=C(76+.75*(bmi-25)+1.3*a+2*inflammation-1.2*activity+5*z("dbp"),55,110)
        P(4079,dbp,"clip(76+0.75(BMI−25)+1.3 a+2 inflammation−1.2 activity+5 Z,55,110)")
        P(4080,dbp+C(41+3*a+.30*(bmi-25)+5*z("pulsepressure"),25,72),"diastolic+clip(41+3 a+0.30(BMI−25)+5 Z,25,72)")
        P(102,C(70-3*activity+2*current+1.5*inflammation+6*z("pulse"),45,110),"clip(70−3 activity+2 current+1.5 inflammation+6 Z,45,110)")
        fvc=C(3.6+.055*(height-165)+.3*sex-.035*(age-55)+.12*activity-.006*pack+.3*z("fvc"),1.5,6.4)
        center=.80-.0015*(age-55); ratio=C(center-.0007*pack+.035*z("ratio"),.55,.94)
        P(3062,fvc,"clip(3.6+0.055(height−165)+0.3 sex−0.035(age−55)+0.12 activity−0.006 pack_years+0.3 Z,1.5,6.4)")
        P(3063,fvc*ratio,"FVC × clip(0.80−0.0015(age−55)−0.0007 pack_years+0.035 Z,0.55,0.94)")
        P(20258,C((ratio-center)/.065,-3.687,4.931),"clip([FEV1/FVC−(0.80−0.0015(age−55))]/0.065,−3.687,4.931); synthetic reference Z")
        P(3064,C(100*fvc*(ratio/.8)*np.exp(.08*z("pef")),100,850),"clip(100 FVC × (ratio/0.8) × exp(0.08 Z),100,850) L/min")

        # Continuous blood count identities retain precision until export.
        iron=.25*diet+.8*z("iron"); turnover=z("redcellturnover")
        mcv=C(90+1.5*alcohol_score-.8*iron+2.8*z("mcv"),76,104)
        mchc=C(33+.30*iron+.55*z("mchc"),30,35)
        rbc=C(4.35+.5*sex+.12*current+.2*z("rbc"),3.2,np.minimum(6.2,np.minimum(550/mcv,18800/(mcv*mchc))))
        mch=mcv*mchc/100
        for fid,value,rule in [(30010,rbc,"clip(4.35+0.5 sex+0.12 current+0.2 Z,3.2,min(6.2,550/MCV,18800/(MCV × MCHC)))"),(30040,mcv,"clip(90+1.5 alcohol_score−0.8 iron+2.8 Z,76,104)"),(30060,mchc,"clip(33+0.30 iron+0.55 Z,30,35)"),(30050,mch,"MCV × MCHC/100"),(30020,rbc*mch/10,"RBC × MCH/10"),(30030,rbc*mcv/10,"RBC × MCV/10"),(30070,C(13.2-.25*iron+.4*np.abs(turnover)+.4*z("rdw"),11.4,18),"clip(13.2−0.25 iron+0.4 |turnover|+0.4 Z,11.4,18)")]: P(fid,value,rule)
        wbc=C(6.3*np.exp(.12*current+.18*inflammation+.15*z("wbc")),2,15)
        neut=.35+.43*S(.3+.45*inflammation-.3*activity+.55*z("neut"))
        lymph=(1-neut)*(.55+.30*S(.3*z("lymph")-.1*a))
        mono=(1-neut-lymph)*(.60+.12*S(.4*z("mono")))
        eos=(1-neut-lymph-mono)*(.70+.20*S(z("eos")))
        baso=1-neut-lymph-mono-eos
        P(30000,wbc,"clip(6.3 exp(0.12 current+0.18 inflammation+0.15 Z),2,15)")
        for cid,pid,fraction,label in [(30140,30200,neut,"neutrophil"),(30120,30180,lymph,"lymphocyte"),(30130,30190,mono,"monocyte"),(30150,30210,eos,"eosinophil"),(30160,30220,baso,"basophil")]:
            P(cid,wbc*fraction,f"WBC × {label}_fraction; exact stick-breaking fractions in MODEL_SPEC")
            P(pid,100*fraction,f"100 × {label}_fraction")
        platelets=C(250*np.exp(-.10*sex+.08*inflammation+.15*z("platelets")),110,450)
        mpv=C(10*np.exp(-.12*np.log(platelets/250)+.055*z("mpv")),8,12)
        P(30080,platelets,"clip(250 exp(−0.10 sex+0.08 inflammation+0.15 Z),110,450)")
        P(30100,mpv,"clip(10 exp(−0.12 ln(platelets/250)+0.055 Z),8,12)")
        P(30090,platelets*mpv/10000,"platelet_count × MPV/10000")
        P(30110,C(17+.3*(mpv-10)+.2*z("pdw"),15.1,19.5),"clip(17+0.3(MPV−10)+0.2 Z,15.1,19.5)")
        retic=C(1.2*np.exp(.20*turnover+.18*z("retic")),.4,3)
        immature=.08+.15*S(turnover+.2*z("immature"))
        scatter=retic*immature*.6
        P(30240,retic,"clip(1.2 exp(0.20 turnover+0.18 Z),0.4,3)")
        P(30250,rbc*retic/100,"RBC × reticulocyte_percentage/100")
        P(30280,immature,"0.08+0.15 σ(turnover+0.2 Z)")
        P(30290,scatter,"reticulocyte_percentage × immature_reticulocyte_fraction × 0.6")
        P(30300,rbc*scatter/100,"RBC × high_scatter_percentage/100")
        P(30260,C(mcv+12+2*turnover+2*z("reticvolume"),82,130),"clip(MCV+12+2 turnover+2 Z,82,130)")
        P(30270,C(.96*mcv+1.2*z("sphered"),65,115),"clip(0.96 MCV+1.2 Z,65,115)")
        nrbc=B("nrbcpresent",.015)*C(.004*np.exp(.45*z("nrbc")),.0005,.03)
        P(30170,nrbc,"Bernoulli(0.015) × clip(0.004 exp(0.45 Z),0.0005,0.03); 10^9/L, continuous despite display-decimal recommendation")
        P(30230,100*nrbc/wbc,"100 nucleated_RBC_count/WBC; assumed denominator WBC")

        lipid,lipid2,hepatic=z("lipid"),z("lipidstorage"),z("hepatic")
        ldl=C(2.8*np.exp(.13*adip+.16*lipid+.08*z("ldl")),1.45,5.8)
        hdl=C(1.5*np.exp(.14*(1-sex)-.14*adip+.09*activity+.10*z("hdl")),.7,2.6)
        tg=C(1.3*np.exp(.23*adip+.18*alcohol_score+.25*lipid2+.20*z("tg")),.4,4.2)
        P(30780,ldl,"clip(2.8 exp(0.13 adip+0.16 lipid+0.08 Z),1.45,5.8)")
        P(30760,hdl,"clip(1.5 exp(0.14(1−sex)−0.14 adip+0.09 activity+0.10 Z),0.7,2.6)")
        P(30870,tg,"clip(1.3 exp(0.23 adip+0.18 alcohol_score+0.25 lipid_storage+0.20 Z),0.4,4.2)")
        P(30690,ldl+hdl+tg/2.2,"LDL+HDL+triglycerides/2.2; idealized assumed lipid accounting")
        P(30640,C(.23+.24*ldl+.035*tg+.045*z("apob"),.412,1.966),"clip(0.23+0.24 LDL+0.035 triglycerides+0.045 Z,0.412,1.966)")
        P(30630,C(.60+.65*hdl+.06*z("apoa"),.771,2.498),"clip(0.60+0.65 HDL+0.06 Z,0.771,2.498)")
        filtration=C(100-.7*(age-45)+10*z("filtration")-3*adip,45,140)
        P(30700,C(75*(1+.22*sex)*np.exp(.08*muscle+.06*z("creatinine"))/(filtration/95),25,260),"clip(75(1+0.22 sex) exp(0.08 muscle+0.06 Z)/(filtration/95),25,260)")
        P(30720,C(.85*(95/filtration)**.8*np.exp(.07*z("cystatin")+.03*adip),.4,2.5),"clip(0.85(95/filtration)^0.8 exp(0.07 Z+0.03 adip),0.4,2.5)")
        P(30670,C(5*(95/filtration)**.45*np.exp(.15*appetite-.08*hydration+.12*z("urea")),2,15),"clip(5(95/filtration)^0.45 exp(0.15 appetite−0.08 hydration+0.12 Z),2,15)")
        P(30880,C(300+45*sex+20*adip+15*alcohol_score+.55*(100-filtration)+20*z("urate"),140,600),"clip(300+45 sex+20 adip+15 alcohol_score+0.55(100−filtration)+20 Z,140,600)")
        P(30620,C(20*np.exp(.25*adip+.15*alcohol_score+.20*hepatic+.18*z("alt")),5,180),"clip(20 exp(0.25 adip+0.15 alcohol_score+0.20 hepatic+0.18 Z),5,180)")
        P(30650,C(20*np.exp(.10*adip+.12*alcohol_score+.18*hepatic+.10*muscle+.16*z("ast")),9,170),"clip(20 exp(0.10 adip+0.12 alcohol_score+0.18 hepatic+0.10 muscle+0.16 Z),9,170)")
        P(30730,C(25*np.exp(.20*adip+.35*alcohol_score+.20*hepatic+.25*z("ggt")),6,300),"clip(25 exp(0.20 adip+0.35 alcohol_score+0.20 hepatic+0.25 Z),6,300)")
        albumin=C(43-.6*a-.8*inflammation-.7*hydration+1.2*z("albumin"),35,50)
        P(30600,albumin,"clip(43−0.6 a−0.8 inflammation−0.7 hydration+1.2 Z,35,50)")
        P(30860,albumin+C(25+1.1*inflammation+z("globulin"),20,32),"albumin+clip(25+1.1 inflammation+Z,20,32)")
        P(30680,C(2.32+.02*(albumin-43)+.035*z("calcium"),2.02,2.65),"clip(2.32+0.02(albumin−43)+0.035 Z,2.02,2.65)")
        P(30810,C(1.1-.04*sex+.04*bone+.09*z("phosphate"),.65,1.6),"clip(1.1−0.04 sex+0.04 bone+0.09 Z,0.65,1.6)")
        P(30610,C(75*np.exp(.10*a+.12*bone+.08*hepatic+.15*z("alp")),35,170),"clip(75 exp(0.10 a+0.12 bone+0.08 hepatic+0.15 Z),35,170)")
        P(30710,C(1.2*np.exp(.70*inflammation+.30*adip+.10*current),.08,50),"clip(1.2 exp(0.70 inflammation+0.30 adip+0.10 current),0.08,50)")
        bilirubin=C(10*np.exp(.35*z("bilirubinclearance")+.12*z("bilirubin")),3.5,40)
        P(30840,bilirubin,"clip(10 exp(0.35 clearance+0.12 Z),3.5,40)")
        P(30660,C(1+.13*bilirubin+.2*z("directbilirubin"),1,.4*bilirubin),"clip(1+0.13 total_bilirubin+0.2 Z,1,0.4 total_bilirubin)")
        glycemic=.30*adip+.13*a+.50*z("glycemic"); long_glucose=5.2*np.exp(.10*glycemic)
        P(30740,C(long_glucose+.35*z("glucose"),3.2,12),"clip(5.2 exp(0.10 glycemic)+0.35 Z,3.2,12)")
        P(30750,C(34+9*(long_glucose-5.2)+2*z("hba1c")-1.2*turnover,18,100),"clip(34+9(long_term_glucose−5.2)+2 Z−1.2 turnover,18,100)")
        P(30770,C(22*np.exp(-.11*a+.10*diet+.15*z("growthaxis")+.12*z("igf")),8,45),"clip(22 exp(−0.11 a+0.10 diet+0.15 growth_axis+0.12 Z),8,45)")
        P(30790,C(30*np.exp(.90*z("heritablelpa")+.15*z("lpa")),3.8,189),"clip(30 exp(0.90 heritable_LpA+0.15 Z),3.8,189)")
        hormone=z("hormone"); transition=B("reproductive_transition",S((age-50)/3))
        estradiol=np.where(sex==1,85*np.exp(.22*z("estradiolmale")),np.where(transition==1,80*np.exp(.30*z("estradiolpost")),480*np.exp(.4*z("cyclephase")+.20*z("estradiolpre"))))
        P(30800,np.minimum(estradiol,12647.8),"male:85 exp(0.22 Z); female transitioned:80 exp(0.30 Z), otherwise480 exp(0.4 phase+0.20 Z); transition∼Bernoulli(σ((age−50)/3)); values<175 censored")
        self.mask(30800,estradiol<175,"Assumed lower assay reporting limit175 pmol/L from dictionary; lower concentrations unreported")
        testosterone=np.where(sex==1,C(14*np.exp(-.13*a-.15*adip+.18*hormone+.16*z("testomale")),2,38),C(1.1*np.exp(-.04*a+.18*hormone+.20*z("testofemale")),.35,3.5))
        P(30850,testosterone,"male:clip(14 exp(−0.13 a−0.15 adip+0.18 hormone+0.16 Z),2,38); female:clip(1.1 exp(−0.04 a+0.18 hormone+0.20 Z),0.35,3.5)")
        P(30830,C(35*np.exp(.45*(1-sex)+.14*a-.17*adip+.15*hormone+.20*z("shbg")),1,200),"clip(35 exp(0.45(1−sex)+0.14 a−0.17 adip+0.15 hormone+0.20 Z),1,200)")
        rf=np.minimum(np.exp(np.log(8)+.65*z("rf")+.12*a+.15*inflammation),118.6)
        P(30820,rf,"min(exp(ln8+0.65 Z+0.12 a+0.15 inflammation),118.6); values<10 censored")
        self.mask(30820,rf<10,"Assumed lower assay reporting limit10 IU/mL from dictionary")
        vitamin=np.minimum(45+5*summer-3*(skin-1)+4*diet+12*z("season")+8*z("vitamin"),169)
        P(30890,vitamin,"min(45+5 summer−3(skin_code−1)+4 diet+12 season+8 Z,169); values<10 censored")
        self.mask(30890,vitamin<10,"Assumed lower assay reporting limit10 nmol/L from dictionary")

        if set(self.x)!=set(self.rows):
            raise ValueError(f"Unimplemented IDs: {sorted(set(self.rows)-set(self.x))}")
        if apply_missingness:
            visit=u("visitmissing")<.005; blood=u("bloodmissing")<.025; survey=u("surveymissing")<.010
            impedance=u("impedancemissing")<.018
            for fid,row in self.rows.items():
                if fid in {"21022","31"}: continue
                if row["Category"] in {"Blood count","Blood biochemistry"}:
                    self.mask(fid,blood,"Shared blood-sample missingness Bernoulli(0.025)")
                    self.mask(fid,u("assaymissing"+fid)<.008,"Independent assay missingness Bernoulli(0.008)")
                elif row["Category"]=="Physical measures":
                    self.mask(fid,visit,"Shared physical-visit missingness Bernoulli(0.005)")
                    if row["Detail_Category"]=="Body composition by impedance":
                        self.mask(fid,impedance,"Shared impedance-session missingness Bernoulli(0.018)")
                    else:
                        self.mask(fid,u("physicalmissing"+fid)<.006,"Independent physical-item missingness Bernoulli(0.006)")
                else:
                    self.mask(fid,survey,"Shared questionnaire missingness Bernoulli(0.010)")
                    self.mask(fid,u("surveymissing"+fid)<(.015+.01*S(-access)),"Per-item questionnaire missingness probability0.015+0.01 σ(−access)")
            for fid in [2139,2149,2159]:
                self.mask(fid,u("sensitivemissing"+str(fid))<.08,"Additional sensitive-item nonresponse Bernoulli(0.08)")
        return np.column_stack([self.x[r["Field_ID"]] for r in self.schema]).astype(np.float32)
    def rule_table(self):
        return [{"Field_ID":r["Field_ID"],"Description":r["Description"],"Unit":r["Unit"],
                 "generation_rule":self.rules[r["Field_ID"]],
                 "missingness_assumption":"; ".join(self.missing.get(r["Field_ID"],["None"]))} for r in self.schema]

def validate(x,schema):
    if x.shape[1]!=len(schema) or x.dtype!=np.float32 or np.isinf(x).any():
        raise AssertionError("Wrong shape, dtype or infinity")
    for j,r in enumerate(schema):
        v=x[:,j]; v=v[np.isfinite(v)]
        lo,hi=float(r["Recommended_Hard_Min"]),float(r["Recommended_Hard_Max"])
        tol=2e-6*max(1,abs(lo),abs(hi))
        if np.any(v<lo-tol) or np.any(v>hi+tol):
            raise AssertionError(f"Range {r['Field_ID']}: {v.min()}..{v.max()}, expected{lo}..{hi}")
        if r["Recommended_Allowed_Values"] and not np.isin(v,json.loads(r["Recommended_Allowed_Values"])).all():
            raise AssertionError(f"Invalid category {r['Field_ID']}")
        if r["Recommended_Data_Type"] in {"bounded_integer_or_count","categorical_integer_code","binary_integer_code"} and not np.equal(v,np.rint(v)).all():
            raise AssertionError(f"Noninteger code/count {r['Field_ID']}")

def source_hashes(base):
    names=["generate.py","test_generator.py","MODEL_SPEC.md","measurement_rules.csv","CONSTRUCTION_RECORD.json","measurement_dictionary.csv","input_manifest.json","TEST_REPORT.json"]
    return {name:sha(base/name) for name in names if (base/name).is_file()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schema",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--n",type=int,default=300000); p.add_argument("--seed",type=int,default=20260831)
    args=p.parse_args()
    if args.n<1: p.error("--n must be positive")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    schema=read_schema(args.schema); model=Population(schema,args.n,args.seed)
    x=model.generate(); validate(x,schema)
    np.save(args.output_dir/"synthetic_features_float32.npy",x,allow_pickle=False)
    (args.output_dir/"feature_columns.json").write_text(json.dumps([r["Field_ID"] for r in schema])+"\n")
    import scipy
    params={"model_version":VERSION,"seed":args.seed,"n":args.n,"n_features":len(schema),"dtype":"float32",
        "model_request":MODEL_REQUEST,"input_sha256":{str(args.schema.resolve()):sha(args.schema),"input_manifest.json":sha(Path(__file__).resolve().parent/"input_manifest.json")},
        "source_sha256":source_hashes(Path(__file__).resolve().parent),
        "feature_order":"Exact supplied CSV row order; string Field_IDs",
        "rng":"Named SHA256(version|seed|stream)-keyed NumPy PCG64, inverse-Gaussian-CDF normals; no sample fitting",
        "determinism":"Fixed seed/code/numerical environment; row prefixes invariant to n",
        "missing_value":"NaN; explicit structural, assay-censoring, block and item assumptions",
        "scope":"Synthetic baseline measurements only, no diagnosis identities, targets, future outcomes or real participants",
        "provenance":"Only supplied measurement dictionary and input manifest read as project inputs. All priors, equations, covariance mechanisms, code semantics and missingness authored as assumptions. No retrieval, empirical statistics or fitting.",
        "parameters_and_rules":model.rule_table(),
        "numerical_environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__}}
    params["output_sha256"]={name:sha(args.output_dir/name) for name in ["synthetic_features_float32.npy","feature_columns.json"]}
    (args.output_dir/"generation_parameters.json").write_text(json.dumps(params,indent=2)+"\n")
    print(json.dumps({"output_dir":str(args.output_dir),"shape":list(x.shape),"seed":args.seed,"validation":"passed"}))
if __name__=="__main__":
    main()
