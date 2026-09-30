#!/usr/bin/env python3
"""
NIFTY NEW RULES-ONLY TREND ENGINE R4
====================================

Standalone, live-causal strategy engine.

Input: raw NIFTY 1-minute OHLC CSV/ZIP only.
Output: signals/trades generated sequentially from rules in this file.

Integrity standard:
- no historical trade rows or timestamps embedded;
- no historical P&L ledger or precomputed signal table;
- no base64/gzip/encoded historical trades;
- no result-forcing checksum or target P&L constant;
- decisions use completed bars and next valid 1-minute fills;
- every paid initial stop is at least 10 NIFTY points;
- R4 preserves R3 and adds a causal CORE stale-risk exit: after a configurable
  number of completed market minutes, if a leg has failed to prove enough MFE
  and remains weak, exit only at the NEXT 1-minute open. No future data is used.

Historical data is used only when supplied at runtime as the raw 1-minute input.
"""
import pandas as pd, numpy as np, itertools, json, math, time, zipfile
from multiprocessing import Pool, cpu_count
DATA=None
BASE={
'gap_exp':1.06,'bull_dist':12.0,'bear_body':0.10,'bear_body_max':None,'buffer':5.0,
'b1_rec':185.0,'b1_wait':225,'b1_gap':9.0,
'b2_rec':90.0,'b2_wait':105,'b2_gap':15.0,
'b3_rec':157.5,'b3_wait':60,'b3_gap':10.0,
'b4_rec':115.0,'b4_wait':60,'b4_gap':0.0,
's1_rec':185.0,'s1_wait':330,'s1_gap':9.0,
's2_rec':71.0,'s2_wait':120,'s2_gap':30.0,
'd_minbars':6,'d_absgap':8.5,'d_norm':0.22,'d_dist':22.5,'d_ratio':1.0,
'dr_rec':15.0,'dr_wait':240,'dr_gap':20.0,'dr_body':0.50,'dr_maxdist':55.0,
# delayed bull defaults disabled
'ub_on':False,'ub_minbars':6,'ub_absgap':8.5,'ub_norm':0.22,'ub_dist':22.5,'ub_ratio':1.0,
'ub_maxre':0,'ubr_rec':20.0,'ubr_wait':240,'ubr_gap':20.0,'ubr_body':0.50,'ubr_maxdist':55.0,
'ub_ret6min':-1e9,'ub_ret3max':1e9,'ub_maxdist':1e9,
# optional rejected-BULL fresh-week continuation monitor
'ubw_on':False,'ubw_minbars':5,'ubw_gap':6.0,'ubw_norm':0.22,'ubw_ratio':1.0,'ubw_dist':12.0,'ubw_maxdist':100.0,'ubw_ret6':20.0,'ubw_n':4,'ubw_body':0.40,
'ubk_maxre':0,'ubkr_rec':40.0,'ubkr_wait':120,'ubkr_gap':15.0,'ubkr_body':0.40,'ubkr_maxdist':80.0,
# optional rejected-BULL pullback -> reclaim monitor (V10 research)
'ubr2_on':False,'ubr2_minbars':8,'ubr2_maxbars':999,'ubr2_max_clock_min':None,'ubr2_arm_dist':-10.0,'ubr2_norm':0.05,'ubr2_dist':5.0,'ubr2_maxdist':30.0,'ubr2_ret6':40.0,'ubr2_body':0.20,
# optional one-shot BEAR post-stop continuation monitor
'psb_on':False,'psb_minbars':4,'psb_gap':6.0,'psb_norm':0.5,'psb_ratio':1.0,'psb_dist':12.0,'psb_maxdist':120.0,'psb_ret6':20.0,'psb_n':6,'psb_body':0.5,
# optional one-shot BULL post-stop continuation monitor
'psu_on':False,'psu_minbars':4,'psu_gap':6.0,'psu_norm':0.5,'psu_ratio':1.0,'psu_dist':12.0,'psu_maxdist':120.0,'psu_ret6':20.0,'psu_n':6,'psu_body':0.5,
# module-specific first-attempt stop widths; re-entry attempts remain 10 unless separately changed
'normal_sl':10.0,'db_sl':10.0,'ub_sl':10.0,'ubk_sl':10.0,'ubw_sl':10.0,'qim_sl':10.0,
# optional one-shot medium BULL continuation after MAIN stop, before ordinary RE1
'mb_on':False,'mb_wait':60,'mb_rec':10.0,'mb_gap':6.0,'mb_norm':0.50,'mb_ratio':1.0,'mb_maxdist':60.0,'mb_body':0.60,'mb_ret3':0.0,'mb_ret6':0.0,
'qim_dyn_sl':False,'qim_sl_floor':10.0,'qim_sl_cap':12.0,'qim_sl_range_mult':0.30,
'db_dyn_sl':False,'db_sl_cap':15.0,'db_sl_range_mult':0.30,
'ub_dyn_sl':False,'ub_sl_cap':15.0,'ub_sl_range_mult':0.30,
'ubk_dyn_sl':False,'ubk_sl_cap':15.0,'ubk_sl_range_mult':0.30,
'ubw_dyn_sl':False,'ubw_sl_cap':15.0,'ubw_sl_range_mult':0.30,
# R4 causal stale-risk guard for CORE legs; disabled by default in BASE.
'core_stale_on':False,'core_stale_minutes':90,'core_stale_mfe':20.0,'core_stale_max_pnl':0.0,
'core_stale_types':('normal','bear_delayed','bull_breakout','bull_quiet_momentum'),
}

def ema(a,span):
    alpha=2/(span+1); out=np.empty(len(a)); out[0]=a[0]
    for i in range(1,len(a)): out[i]=alpha*a[i]+(1-alpha)*out[i-1]
    return out

def load(path):
    path=str(path)
    if path.lower().endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/') and not n.split('/')[-1].startswith('._')]
            if len(names)!=1:
                raise ValueError(f'Expected exactly one CSV inside ZIP, found: {names}')
            with z.open(names[0]) as f:
                raw=pd.read_csv(f)
    else:
        raw=pd.read_csv(path)
    raw.columns=[c.lower().strip() for c in raw.columns]
    x=raw[['timestamp','open','high','low','close']].copy(); x.timestamp=pd.to_datetime(x.timestamp)
    for c in ['open','high','low','close']: x[c]=pd.to_numeric(x[c])
    x=x.sort_values('timestamp').drop_duplicates('timestamp',keep='last').reset_index(drop=True)
    # LIVE-STRICT: execute only in the regular NIFTY session. This prevents the
    # next raw row after a 15:15 signal from becoming a Muhurat/evening row.
    mins=x.timestamp.dt.hour*60+x.timestamp.dt.minute
    x=x[(mins>=555)&(mins<930)].reset_index(drop=True)
    return x

def bars30(df):
    full=df.copy(); full['orig_idx']=np.arange(len(full));dt=pd.DatetimeIndex(full.timestamp);mins=dt.hour*60+dt.minute
    s=full[(mins>=555)&(mins<930)].copy();dt2=pd.DatetimeIndex(s.timestamp);mins2=dt2.hour*60+dt2.minute;s['day']=dt2.normalize();s['bucket']=((mins2-555)//30).astype(int)
    b=s.groupby(['day','bucket'],sort=False).agg(last_idx=('orig_idx','last'),bar_start=('timestamp','first'),open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last')).reset_index(drop=True)
    b=b[b.last_idx+1<len(full)].reset_index(drop=True);c=b.close.to_numpy(float);ef=ema(c,19);es=ema(c,29);gap=np.abs(ef-es);rng=(b.high-b.low).to_numpy(float)
    b['ema_fast']=ef;b['ema_slow']=es;b['ema_gap']=gap;b['avg6']=pd.Series(rng).shift(1).rolling(6).mean().to_numpy(float)
    b['bull_cross']=(ef>es)&np.r_[False,ef[:-1]<=es[:-1]];b['bear_cross']=(ef<es)&np.r_[False,ef[:-1]>=es[:-1]]
    return b
DF=None;B=None

def init(path):
    global DF,B,DATA
    DATA=str(path)
    DF=load(DATA)
    B=bars30(DF)

def metrics(a):
    a=np.asarray(a,float)
    if not len(a):return 0.,0.,np.nan,np.nan
    eq=np.cumsum(a);pk=np.maximum.accumulate(np.r_[0.,eq]);dd=abs((np.r_[0.,eq]-pk).min());gp=a[a>0].sum();gl=-a[a<0].sum();return float(a.sum()),float(dd),float(gp/gl if gl else np.inf),float((a>0).mean()*100)
def lastbar(last,k):return int(np.searchsorted(last,k,side='right')-1)

def sim(p,frames=False):
    df,b=DF,B;O=df.open.to_numpy(float);H=df.high.to_numpy(float);L=df.low.to_numpy(float);times=df.timestamp.to_numpy('datetime64[ns]');last=b.last_idx.to_numpy(np.int64);ef=b.ema_fast.to_numpy(float);es=b.ema_slow.to_numpy(float);gap=b.ema_gap.to_numpy(float);avg6=b.avg6.to_numpy(float)
    sig=np.flatnonzero(b.bull_cross.to_numpy(bool)|b.bear_cross.to_numpy(bool));dirs=np.where(b.loc[sig,'bull_cross'].to_numpy(bool),1,-1)
    masters=[];legs=[];mid=0;week_entries={};week_last={}
    bull_rules=[(p['b1_rec'],p['b1_wait'],p['b1_gap']),(p['b2_rec'],p['b2_wait'],p['b2_gap']),(p['b3_rec'],p['b3_wait'],p['b3_gap']),(p['b4_rec'],p['b4_wait'],p['b4_gap'])]
    bear_rules=[(p['s1_rec'],p['s1_wait'],p['s1_gap']),(p['s2_rec'],p['s2_wait'],p['s2_gap'])]
    for j in range(len(sig)-1):
        bi=int(sig[j]);bn=int(sig[j+1]);base=int(last[bi]+1);xi=int(last[bn]+1)
        if xi<=base or xi>=len(df):continue
        if times[base]<np.datetime64(p['start_date']) or times[base]>=np.datetime64(p['end_date']):continue
        d=int(dirs[j]);ok=False;delayed=False;dtype='normal';sig_wk=pd.Timestamp(b.at[bi,'bar_start']).to_period('W-SUN').start_time;sig_quiet=(week_entries.get(sig_wk,0)==0)
        if bi>=1 and gap[bi-1]>0 and gap[bi]/gap[bi-1]>=p['gap_exp']:
            c=float(b.at[bi,'close'])
            if d==1:ok=min(c-ef[bi],c-es[bi])/c*10000>=p['bull_dist']
            else:
                rng=float(b.at[bi,'high']-b.at[bi,'low']);body=abs(float(b.at[bi,'close']-b.at[bi,'open']));bratio=body/rng if rng>0 else -999
                ok=rng>0 and bratio>=p['bear_body'] and (p.get('bear_body_max') is None or bratio<=p.get('bear_body_max'))
        # causal quiet-week strong-momentum BULL exception at the crossover itself
        if (not ok) and d==1 and sig_quiet and p.get('qim_on',False) and bi>=6:
            c=float(b.at[bi,'close']);oo=float(b.at[bi,'open']);hi=float(b.at[bi,'high']);lo=float(b.at[bi,'low'])
            dist=min(c-ef[bi],c-es[bi])/c*10000;ret6=(c/float(b.at[bi-6,'close'])-1)*10000
            br=hi-lo;body=(c-oo)/br if br>0 else -999
            norm=gap[bi]/avg6[bi] if np.isfinite(avg6[bi]) and avg6[bi]>0 else -999
            if ret6>=p.get('qim_ret6',80) and dist>=p.get('qim_dist',40) and body>=p.get('qim_body',0) and norm>=p.get('qim_norm',0):
                ok=True;dtype='bull_quiet_momentum'
        if ok:entrybar=bi
        else:
            entrybar=-1
            if d==-1:
                for q in range(bi+int(p['d_minbars']),bn):
                    if not ef[q]<es[q] or gap[q]<p['d_absgap']:continue
                    if not np.isfinite(avg6[q]) or avg6[q]<=0 or gap[q]/avg6[q]<p['d_norm']:continue
                    if q<1 or gap[q-1]<=0 or gap[q]/gap[q-1]<p['d_ratio']:continue
                    c=float(b.at[q,'close']);bps=min(ef[q]-c,es[q]-c)/c*10000
                    if bps<p['d_dist']:continue
                    if q>=6 and p.get('d_ret6min',-1e9)>-1e8:
                        r6d=(float(b.at[q-6,'close'])/c-1)*10000
                        if r6d<p.get('d_ret6min',-1e9): continue
                    if q>=3 and p.get('d_ret3min',-1e9)>-1e8:
                        r3d=(float(b.at[q-3,'close'])/c-1)*10000
                        if r3d<p.get('d_ret3min',-1e9): continue
                    entrybar=q;delayed=True;dtype='bear_delayed';break
            elif p.get('ub_on',False) or p.get('ubk_on',False) or p.get('ubw_on',False) or p.get('ubr2_on',False):
                # LIVE-CAUSAL competition between the existing primary route delayed-BULL
                # module and the optional rejected-BULL breakout fallback.
                # At each completed 30m bar, the primary continuation route gets first priority; if it does
                # not trigger on THAT BAR, the breakout rule may trigger on the
                # same completed bar. We never look ahead to see whether primary route
                # would trigger on a later bar before allowing the fallback.
                starts=[]
                if p.get('ub_on',False): starts.append(bi+int(p['ub_minbars']))
                if p.get('ubk_on',False): starts.append(bi+int(p.get('ubk_minbars',5)))
                if p.get('ubw_on',False): starts.append(bi+int(p.get('ubw_minbars',5)))
                # Reclaim needs to observe the pullback/reset from the first
                # completed bar after crossover, so its live state starts at bi+1.
                if p.get('ubr2_on',False): starts.append(bi+1)
                q0=max(bi+1, min(starts)) if starts else bn
                ubr2_armed=False
                for q in range(q0,bn):
                    # Existing primary route delayed-BULL rule, evaluated first on q.
                    if p.get('ub_on',False) and q>=bi+int(p['ub_minbars']):
                        pass_ub=True
                        if not ef[q]>es[q] or gap[q]<p['ub_absgap']: pass_ub=False
                        if pass_ub and (not np.isfinite(avg6[q]) or avg6[q]<=0 or gap[q]/avg6[q]<p['ub_norm']): pass_ub=False
                        if pass_ub and (q<1 or gap[q-1]<=0 or gap[q]/gap[q-1]<p['ub_ratio']): pass_ub=False
                        if pass_ub:
                            c=float(b.at[q,'close']);bps=min(c-ef[q],c-es[q])/c*10000
                            if bps<p['ub_dist'] or bps>p['ub_maxdist']: pass_ub=False
                        if pass_ub and q<6: pass_ub=False
                        if pass_ub:
                            c=float(b.at[q,'close']);c3=float(b.at[q-3,'close']);c6=float(b.at[q-6,'close'])
                            r3=(c/c3-1)*10000; r6=(c/c6-1)*10000
                            if r6<p['ub_ret6min'] or r3>p['ub_ret3max']: pass_ub=False
                        if pass_ub:
                            entrybar=q;delayed=True;dtype='bull_delayed';break

                    # Optional causal breakout fallback, evaluated only if primary route
                    # did not trigger on this same completed bar. IMPORTANT: a
                    # failed breakout check must NOT skip later fallback modules.
                    if p.get('ubk_on',False) and q>=bi+int(p.get('ubk_minbars',5)):
                        pass_ubk=True
                        n=int(p.get('ubk_n',4))
                        if q<n: pass_ubk=False
                        if pass_ubk and (not ef[q]>es[q] or gap[q]<p.get('ubk_gap',6.0)): pass_ubk=False
                        if pass_ubk and (not np.isfinite(avg6[q]) or avg6[q]<=0 or gap[q]/avg6[q]<p.get('ubk_norm',0.22)): pass_ubk=False
                        if pass_ubk and (q<1 or gap[q-1]<=0 or gap[q]/gap[q-1]<p.get('ubk_ratio',1.0)): pass_ubk=False
                        if pass_ubk:
                            c=float(b.at[q,'close']);oo=float(b.at[q,'open'])
                            dist=min(c-ef[q],c-es[q])/c*10000
                            if dist<p.get('ubk_dist',12.0) or dist>p.get('ubk_maxdist',65.0): pass_ubk=False
                        if pass_ubk and q<6: pass_ubk=False
                        if pass_ubk:
                            r6=(c/float(b.at[q-6,'close'])-1)*10000
                            if r6<p.get('ubk_ret6',0.0): pass_ubk=False
                        if pass_ubk and c<=float(b.loc[q-n:q-1,'high'].max()): pass_ubk=False
                        if pass_ubk:
                            rng=float(b.at[q,'high']-b.at[q,'low']);body=abs(c-oo)
                            if rng<=0 or body/rng<p.get('ubk_body',0.0): pass_ubk=False
                        if pass_ubk:
                            entrybar=q;delayed=True;dtype='bull_breakout';break

                    # Optional fresh-week continuation monitor for a rejected BULL.
                    # Uses ONLY state known at this completed bar: there must be no
                    # accepted master yet in the calendar week of the prospective
                    # next-minute fill. It does not know future weekly activity.
                    if p.get('ubw_on',False) and q>=bi+int(p.get('ubw_minbars',5)):
                        pass_ubw=True
                        n=int(p.get('ubw_n',4))
                        if q<n: pass_ubw=False
                        if pass_ubw:
                            fill_i=int(last[q]+1)
                            if fill_i>=xi: pass_ubw=False
                        if pass_ubw:
                            fill_wk=pd.Timestamp(times[fill_i]).to_period('W-SUN').start_time
                            if week_entries.get(fill_wk,0)!=0: pass_ubw=False
                        if pass_ubw and (not ef[q]>es[q] or gap[q]<p.get('ubw_gap',6.0)): pass_ubw=False
                        if pass_ubw and (not np.isfinite(avg6[q]) or avg6[q]<=0 or gap[q]/avg6[q]<p.get('ubw_norm',0.22)): pass_ubw=False
                        if pass_ubw and (q<1 or gap[q-1]<=0 or gap[q]/gap[q-1]<p.get('ubw_ratio',1.0)): pass_ubw=False
                        if pass_ubw:
                            c=float(b.at[q,'close']);oo=float(b.at[q,'open'])
                            dist=min(c-ef[q],c-es[q])/c*10000
                            if dist<p.get('ubw_dist',12.0) or dist>p.get('ubw_maxdist',100.0): pass_ubw=False
                        if pass_ubw and q<6: pass_ubw=False
                        if pass_ubw:
                            r6=(c/float(b.at[q-6,'close'])-1)*10000
                            if r6<p.get('ubw_ret6',20.0): pass_ubw=False
                        if pass_ubw and c<=float(b.loc[q-n:q-1,'high'].max()): pass_ubw=False
                        if pass_ubw:
                            rng=float(b.at[q,'high']-b.at[q,'low']);body=c-oo
                            if rng<=0 or body/rng<p.get('ubw_body',0.40): pass_ubw=False
                        if pass_ubw:
                            entrybar=q;delayed=True;dtype='bull_weekly_monitor';break


                    # V10: rejected-BULL pullback -> reclaim monitor (strict live-causal).
                    # The reset must have happened on a PRIOR completed 30m bar;
                    # the reclaim is evaluated on a later completed bar and fills
                    # only at the next regular-session 1m open.
                    if p.get('ubr2_on',False):
                        c2=float(b.at[q,'close']); oo2=float(b.at[q,'open'])
                        dist2=min(c2-ef[q],c2-es[q])/c2*10000
                        max_clock=p.get('ubr2_max_clock_min',None)
                        elapsed_clock=(pd.Timestamp(b.at[q,'bar_start'])-pd.Timestamp(b.at[bi,'bar_start'])).total_seconds()/60.0
                        within_time=(q<=bi+int(p.get('ubr2_maxbars',999))) and (max_clock is None or elapsed_clock<=float(max_clock))
                        if (ubr2_armed and q>=bi+int(p.get('ubr2_minbars',8)) and within_time):
                            pass_ubr2=True
                            if not ef[q]>es[q]: pass_ubr2=False
                            if pass_ubr2 and (not np.isfinite(avg6[q]) or avg6[q]<=0 or gap[q]/avg6[q]<p.get('ubr2_norm',0.05)): pass_ubr2=False
                            if pass_ubr2 and (dist2<p.get('ubr2_dist',5.0) or dist2>p.get('ubr2_maxdist',30.0)): pass_ubr2=False
                            if pass_ubr2 and q<6: pass_ubr2=False
                            if pass_ubr2:
                                r6_2=(c2/float(b.at[q-6,'close'])-1)*10000
                                if r6_2<p.get('ubr2_ret6',40.0): pass_ubr2=False
                            if pass_ubr2:
                                rng2=float(b.at[q,'high']-b.at[q,'low']); body2=c2-oo2
                                if rng2<=0 or body2/rng2<p.get('ubr2_body',0.20): pass_ubr2=False
                            if pass_ubr2:
                                entrybar=q;delayed=True;dtype='bull_reclaim';break
                        # Arm only AFTER evaluating q, so reset and reclaim can
                        # never use the same completed candle.
                        if dist2<=p.get('ubr2_arm_dist',-10.0):
                            ubr2_armed=True
            if entrybar<0:continue
        ei=int(last[entrybar]+1)
        if ei>=xi:continue
        entry_wk=pd.Timestamp(times[ei]).to_period('W-SUN').start_time
        # LIVE-CAUSAL weekly chop gate: if the first accepted master of this
        # week has already completed as a loss, skip a second low-confidence
        # fallback master (delayed BEAR or BULL breakout). This uses only
        # realized prior-week-to-date state; no future weekly count is used.
        if p.get('weekly_gate_on',False):
            if week_entries.get(entry_wk,0)==1 and week_last.get(entry_wk,0)<=0 and dtype in ('bear_delayed','bull_breakout'):
                continue
        mid+=1;ep=float(O[ei]);xp=float(O[xi]);natural=d*(xp-ep);cur=ei;curp=ep;re=0;pts=0.;cnt=0
        rules=bull_rules if d==1 else bear_rules;maxre=4 if d==1 else 2
        if dtype=='bear_delayed':maxre=1
        if dtype=='bull_delayed':maxre=int(p['ub_maxre'])
        if dtype=='bull_resumption':maxre=0
        if dtype=='bull_breakout':maxre=int(p.get('ubk_maxre',0))
        if dtype=='bull_quiet_momentum':maxre=0
        if dtype=='bull_weekly_monitor':maxre=0
        if dtype=='bull_reclaim':maxre=0
        while True:
            if re==0:
                erng=float(b.at[entrybar,'high']-b.at[entrybar,'low'])
                slv = ((max(10.0,min(p.get('db_sl_cap',15.0),p.get('db_sl_range_mult',0.30)*erng)) if p.get('db_dyn_sl',False) else p.get('db_sl',10.0)) if dtype=='bear_delayed' else
                       (max(10.0,min(p.get('ub_sl_cap',15.0),p.get('ub_sl_range_mult',0.30)*erng)) if p.get('ub_dyn_sl',False) else p.get('ub_sl',10.0)) if dtype=='bull_delayed' else
                       (max(10.0,min(p.get('ubk_sl_cap',15.0),p.get('ubk_sl_range_mult',0.30)*erng)) if p.get('ubk_dyn_sl',False) else p.get('ubk_sl',10.0)) if dtype=='bull_breakout' else
                       (max(10.0,min(p.get('ubw_sl_cap',15.0),p.get('ubw_sl_range_mult',0.30)*erng)) if p.get('ubw_dyn_sl',False) else p.get('ubw_sl',10.0)) if dtype=='bull_weekly_monitor' else
                       (max(p.get('qim_sl_floor',10.0), min(p.get('qim_sl_cap',12.0), p.get('qim_sl_range_mult',0.30)*erng)) if p.get('qim_dyn_sl',False) else p.get('qim_sl',10.0)) if dtype=='bull_quiet_momentum' else
                       p.get('normal_sl',10.0))
            else:
                slv=10.0
            sl=curp-slv if d==1 else curp+slv;st=False;si=-1;sp=np.nan;stale=False;mfe_leg=0.0
            market_age=0
            for k in range(cur,xi+1):
                if d==1:
                    if O[k]<=sl:st=True;si=k;sp=float(O[k]);break
                    if L[k]<=sl:st=True;si=k;sp=float(sl);break
                    mfe_leg=max(mfe_leg,float(H[k]-curp))
                else:
                    if O[k]>=sl:st=True;si=k;sp=float(O[k]);break
                    if H[k]>=sl:st=True;si=k;sp=float(sl);break
                    mfe_leg=max(mfe_leg,float(curp-L[k]))
                market_age += 1
                # R4 stale-risk decision uses only the just-completed minute k.
                # Fill is strictly the next available regular-session minute open.
                if (p.get('core_stale_on',False) and dtype in p.get('core_stale_types',())
                    and market_age>=int(p.get('core_stale_minutes',90)) and k+1<=xi):
                    close_pnl=float(d*(float(df.at[k,'close'])-curp))
                    if mfe_leg < float(p.get('core_stale_mfe',20.0)) and close_pnl <= float(p.get('core_stale_max_pnl',0.0)):
                        stale=True;si=k+1;sp=float(O[si]);break
            cnt+=1
            if stale:
                pp=d*(sp-curp);pts+=pp;legs.append((mid,re,pp,dtype,pd.Timestamp(times[cur]),pd.Timestamp(times[si]),float(curp),float(sp),'stale_risk'));break
            if not st:
                pp=d*(xp-curp);pts+=pp;legs.append((mid,re,pp,dtype,pd.Timestamp(times[cur]),pd.Timestamp(times[xi]),float(curp),float(xp),'natural'));break
            pp=d*(sp-curp);pts+=pp;legs.append((mid,re,pp,dtype,pd.Timestamp(times[cur]),pd.Timestamp(times[si]),float(curp),float(sp),'stop'))
            if re>=maxre:break
            if dtype=='bear_delayed':rec,wait,gmin=p['dr_rec'],p['dr_wait'],p['dr_gap']
            elif dtype=='bull_delayed':rec,wait,gmin=p['ubr_rec'],p['ubr_wait'],p['ubr_gap']
            elif dtype=='bull_breakout':rec,wait,gmin=p.get('ubkr_rec',40.0),p.get('ubkr_wait',120),p.get('ubkr_gap',15.0)
            else:rec,wait,gmin=rules[re]
            # wait is ELAPSED CLOCK MINUTES, intentionally including overnight/weekends
            earliest=times[si]+np.timedelta64(int(wait),'m');k0=si+1
            while k0<xi and times[k0]<earliest:k0+=1
            trig=sp+rec if d==1 else sp-rec;confirm=trig+p['buffer'] if d==1 else trig-p['buffer']

            # STRICT LIVE-CAUSAL chronological competition for the optional
            # normal-BULL medium continuation versus ordinary RE1. We do NOT
            # precompute a future RE1 timestamp. Starting from the information
            # available after the MAIN stop, walk forward minute by minute. At
            # each completed minute: ordinary RE1 gets first priority if its
            # trigger and latest completed-30m state qualify; otherwise a medium
            # setup can qualify only when a new 30m candle has just completed.
            if p.get('mb_on',False) and d==1 and dtype=='normal' and re==0:
                mearly=times[si]+np.timedelta64(int(p.get('mb_wait',60)),'m')
                mk0=si+1
                while mk0<xi and times[mk0]<mearly: mk0+=1
                scan0=min(k0,mk0);handled=False
                for kk in range(scan0,xi):
                    # Existing ordinary RE1, evaluated using only data through kk.
                    if kk>=k0 and H[kk]>=confirm:
                        bj=lastbar(last,kk)
                        if bj>=1 and gap[bj]>=gmin and gap[bj]>gap[bj-1] and ef[bj]>es[bj]:
                            if kk+1<=xi:
                                cur=kk+1;curp=float(O[cur]);re+=1;handled=True
                            break
                    # Medium setup is a completed-30m decision only.
                    if kk<mk0: continue
                    q=int(np.searchsorted(last,kk,side='left'))
                    if q>=len(last) or int(last[q])!=kk or q<=entrybar: continue
                    fi=kk+1
                    if fi>=xi: continue
                    c=float(b.at[q,'close']);oo=float(b.at[q,'open']);hi=float(b.at[q,'high']);lo=float(b.at[q,'low'])
                    if not ef[q]>es[q]: continue
                    if c-sp < p.get('mb_rec',10.0): continue
                    if gap[q] < p.get('mb_gap',6.0): continue
                    if not np.isfinite(avg6[q]) or avg6[q]<=0 or gap[q]/avg6[q] < p.get('mb_norm',0.50): continue
                    if q<1 or gap[q-1]<=0 or gap[q]/gap[q-1] < p.get('mb_ratio',1.0): continue
                    dist=min(c-ef[q],c-es[q])/c*10000
                    if dist<0 or dist>p.get('mb_maxdist',60.0): continue
                    rng=hi-lo;body=(c-oo)/rng if rng>0 else -999
                    if body < p.get('mb_body',0.60) or q<6: continue
                    r3=(c/float(b.at[q-3,'close'])-1)*10000;r6=(c/float(b.at[q-6,'close'])-1)*10000
                    if r3 < p.get('mb_ret3',0.0) or r6 < p.get('mb_ret6',0.0): continue
                    # Medium fills next minute and carries the same 10-point SL.
                    mp=float(O[fi]);msl=mp-10.0;mst=False;msi=-1;msp=np.nan
                    for jj in range(fi,xi+1):
                        if O[jj]<=msl: mst=True;msi=jj;msp=float(O[jj]);break
                        if L[jj]<=msl: mst=True;msi=jj;msp=float(msl);break
                    cnt+=1
                    if not mst:
                        mpp=float(xp-mp);pts+=mpp;legs.append((mid,99,mpp,'normal_bull_medium',pd.Timestamp(times[fi]),pd.Timestamp(times[xi]),float(mp),float(xp),'natural'));handled=True
                        break
                    mpp=float(msp-mp);pts+=mpp;legs.append((mid,99,mpp,'normal_bull_medium',pd.Timestamp(times[fi]),pd.Timestamp(times[msi]),float(mp),float(msp),'stop'))
                    # Medium stopped. From this now-known stop time, resume the
                    # original RE1 state sequentially; no trigger before msi can
                    # be taken retroactively.
                    found2=-1
                    for jj in range(max(msi+1,k0),xi):
                        if H[jj]<confirm: continue
                        bj2=lastbar(last,jj)
                        if bj2<1 or gap[bj2]<gmin or not(gap[bj2]>gap[bj2-1]) or not ef[bj2]>es[bj2]: continue
                        found2=jj;break
                    if found2>=0 and found2+1<=xi:
                        cur=found2+1;curp=float(O[cur]);re+=1
                    handled=True;break
                # Every path above is final for this iteration: either RE1 was
                # scheduled, medium ran to crossover, medium stopped and RE1 was
                # scheduled, or neither trigger occurred before crossover.
                if handled and re>0: continue
                break

            found=-1
            for k in range(k0,xi):
                if d==1 and H[k]<confirm:continue
                if d==-1 and L[k]>confirm:continue
                bj=lastbar(last,k)
                if bj<1 or gap[bj]<gmin or not(gap[bj]>gap[bj-1]):continue
                if d==1 and not ef[bj]>es[bj]:continue
                if d==-1 and not ef[bj]<es[bj]:continue
                if dtype=='bear_delayed':
                    br=float(b.at[bj,'high']-b.at[bj,'low']);body=abs(float(b.at[bj,'close']-b.at[bj,'open']));c=float(b.at[bj,'close']);dist=min(ef[bj]-c,es[bj]-c)/c*10000
                    if not(br>0 and float(b.at[bj,'close'])<float(b.at[bj,'open']) and body/br>=p['dr_body'] and dist<=p['dr_maxdist']):found=-2
                    else:found=k
                    break
                if dtype=='bull_delayed':
                    br=float(b.at[bj,'high']-b.at[bj,'low']);body=abs(float(b.at[bj,'close']-b.at[bj,'open']));c=float(b.at[bj,'close']);dist=min(c-ef[bj],c-es[bj])/c*10000
                    if not(br>0 and float(b.at[bj,'close'])>float(b.at[bj,'open']) and body/br>=p['ubr_body'] and dist<=p['ubr_maxdist']):found=-2
                    else:found=k
                    break
                if dtype=='bull_breakout':
                    br=float(b.at[bj,'high']-b.at[bj,'low']);body=abs(float(b.at[bj,'close']-b.at[bj,'open']));c=float(b.at[bj,'close']);dist=min(c-ef[bj],c-es[bj])/c*10000
                    if not(br>0 and float(b.at[bj,'close'])>float(b.at[bj,'open']) and body/br>=p.get('ubkr_body',0.40) and dist<=p.get('ubkr_maxdist',80.0)):found=-2
                    else:found=k
                    break
                found=k;break
            # Optional one-shot MEDIUM BULL continuation after the MAIN stop.
            # It competes chronologically with ordinary RE1. It uses only completed
            # 30m bars and fills at the next regular-session 1m open. If it stops,
            # ordinary RE1 resumes from then onward but remains anchored to the
            # original MAIN stop, so we do not retroactively take a RE1 that would
            # have occurred while the medium leg was open.
            micro_taken=False
            if p.get('mb_on',False) and d==1 and dtype=='normal' and re==0:
                # first completed 30m bar after the live wait
                mearly=times[si]+np.timedelta64(int(p.get('mb_wait',60)),'m')
                mk0=si+1
                while mk0<xi and times[mk0]<mearly: mk0+=1
                q0=max(entrybar+1,lastbar(last,mk0)+1)
                std_entry=(found+1) if found>=0 else xi
                mq=-1;mi=-1
                for q in range(q0,bn):
                    fi=int(last[q]+1)
                    if fi>=xi or fi>=std_entry: break
                    c=float(b.at[q,'close']);oo=float(b.at[q,'open']);hi=float(b.at[q,'high']);lo=float(b.at[q,'low'])
                    if not ef[q]>es[q]: continue
                    if c-sp < p.get('mb_rec',10.0): continue
                    if gap[q] < p.get('mb_gap',6.0): continue
                    if not np.isfinite(avg6[q]) or avg6[q]<=0 or gap[q]/avg6[q] < p.get('mb_norm',0.50): continue
                    if q<1 or gap[q-1]<=0 or gap[q]/gap[q-1] < p.get('mb_ratio',1.0): continue
                    dist=min(c-ef[q],c-es[q])/c*10000
                    if dist<0 or dist>p.get('mb_maxdist',60.0): continue
                    rng=hi-lo; body=(c-oo)/rng if rng>0 else -999
                    if body < p.get('mb_body',0.60): continue
                    if q<6: continue
                    r3=(c/float(b.at[q-3,'close'])-1)*10000
                    r6=(c/float(b.at[q-6,'close'])-1)*10000
                    if r3 < p.get('mb_ret3',0.0) or r6 < p.get('mb_ret6',0.0): continue
                    mq=q;mi=fi;break
                if mi>=0:
                    micro_taken=True
                    mp=float(O[mi]); msl=mp-10.0; mst=False;msi=-1;msp=np.nan
                    for kk in range(mi,xi+1):
                        if O[kk]<=msl: mst=True;msi=kk;msp=float(O[kk]);break
                        if L[kk]<=msl: mst=True;msi=kk;msp=float(msl);break
                    cnt+=1
                    if not mst:
                        mpp=float(xp-mp);pts+=mpp;legs.append((mid,99,mpp,'normal_bull_medium',pd.Timestamp(times[mi]),pd.Timestamp(times[xi]),float(mp),float(xp),'natural'))
                        break
                    mpp=float(msp-mp);pts+=mpp;legs.append((mid,99,mpp,'normal_bull_medium',pd.Timestamp(times[mi]),pd.Timestamp(times[msi]),float(mp),float(msp),'stop'))
                    # Re-scan ordinary RE1 from after the medium stop, anchored to
                    # the original MAIN stop (si/sp) and original wait/trigger.
                    k02=max(msi+1,k0); found2=-1
                    for kk in range(k02,xi):
                        if H[kk]<confirm: continue
                        bj=lastbar(last,kk)
                        if bj<1 or gap[bj]<gmin or not(gap[bj]>gap[bj-1]): continue
                        if not ef[bj]>es[bj]: continue
                        found2=kk;break
                    if found2>=0 and found2+1<=xi:
                        cur=found2+1;curp=float(O[cur]);re+=1
                    else:
                        break
            if micro_taken:
                # Either the medium leg ran to the opposite crossover (break above)
                # or we have already scheduled ordinary RE1 after its stop.
                if re>0: continue
                break
            if found>=0 and found+1<=xi:cur=found+1;curp=float(O[cur]);re+=1
            else:break


        # Optional LIVE-CAUSAL BULL post-stop continuation monitor.
        # One extra continuation attempt after the current BULL master has
        # actually stopped out and is flat; all checks use completed bars.
        if p.get('psu_on',False) and d==1 and st and si>=0 and pts<=0 and si+1<xi:
            stop_bj=lastbar(last,si)
            q0=max(stop_bj+int(p.get('psu_minbars',4)), entrybar+1)
            found_q=-1
            for q in range(q0,bn):
                n=int(p.get('psu_n',6))
                if q<n: continue
                fill_i=int(last[q]+1)
                if fill_i<=si or fill_i>=xi: continue
                if not ef[q]>es[q] or gap[q]<p.get('psu_gap',6.0): continue
                if not np.isfinite(avg6[q]) or avg6[q]<=0 or gap[q]/avg6[q]<p.get('psu_norm',0.5): continue
                if q<1 or gap[q-1]<=0 or gap[q]/gap[q-1]<p.get('psu_ratio',1.0): continue
                c=float(b.at[q,'close']); oo=float(b.at[q,'open'])
                dist=min(c-ef[q],c-es[q])/c*10000
                if dist<p.get('psu_dist',12.0) or dist>p.get('psu_maxdist',120.0): continue
                if q<6: continue
                r6=(c/float(b.at[q-6,'close'])-1)*10000
                if r6<p.get('psu_ret6',20.0): continue
                if c<=float(b.loc[q-n:q-1,'high'].max()): continue
                rng=float(b.at[q,'high']-b.at[q,'low']); body=c-oo
                if rng<=0 or body/rng<p.get('psu_body',0.5): continue
                found_q=q; break
            if found_q>=0:
                ri=int(last[found_q]+1); rp=float(O[ri]); sl=rp-10; rst=False; rsi=-1; rsp=np.nan
                for k in range(ri,xi+1):
                    if O[k]<=sl: rst=True; rsi=k; rsp=float(O[k]); break
                    if L[k]<=sl: rst=True; rsi=k; rsp=float(sl); break
                if rst: rpp=(rsp-rp)
                else: rpp=(xp-rp)
                pts+=rpp; cnt+=1; re+=1
                legs.append((mid,re,rpp,'bull_poststop',pd.Timestamp(times[ri]),pd.Timestamp(times[rsi if rst else xi]),float(rp),float(rsp if rst else xp),'stop' if rst else 'natural'))

        # Optional LIVE-CAUSAL BEAR post-stop continuation monitor.
        # Only after this master has actually stopped out and is flat; scan
        # subsequent completed 30m bars until the already-current opposite
        # crossover boundary. One extra attempt only, entered next 1m open.
        if p.get('psb_on',False) and d==-1 and st and si>=0 and pts<=0 and si+1<xi:
            stop_bj=lastbar(last,si)
            q0=max(stop_bj+int(p.get('psb_minbars',4)), entrybar+1)
            found_q=-1
            for q in range(q0,bn):
                n=int(p.get('psb_n',6))
                if q<n: continue
                fill_i=int(last[q]+1)
                if fill_i<=si or fill_i>=xi: continue
                if not ef[q]<es[q] or gap[q]<p.get('psb_gap',6.0): continue
                if not np.isfinite(avg6[q]) or avg6[q]<=0 or gap[q]/avg6[q]<p.get('psb_norm',0.5): continue
                if q<1 or gap[q-1]<=0 or gap[q]/gap[q-1]<p.get('psb_ratio',1.0): continue
                c=float(b.at[q,'close']); oo=float(b.at[q,'open'])
                dist=min(ef[q]-c,es[q]-c)/c*10000
                if dist<p.get('psb_dist',12.0) or dist>p.get('psb_maxdist',120.0): continue
                if q<6: continue
                r6=(float(b.at[q-6,'close'])/c-1)*10000
                if r6<p.get('psb_ret6',20.0): continue
                if c>=float(b.loc[q-n:q-1,'low'].min()): continue
                rng=float(b.at[q,'high']-b.at[q,'low']); body=oo-c
                if rng<=0 or body/rng<p.get('psb_body',0.5): continue
                found_q=q; break
            if found_q>=0:
                ri=int(last[found_q]+1); rp=float(O[ri]); sl=rp+10; rst=False; rsi=-1; rsp=np.nan
                for k in range(ri,xi+1):
                    if O[k]>=sl: rst=True; rsi=k; rsp=float(O[k]); break
                    if H[k]>=sl: rst=True; rsi=k; rsp=float(sl); break
                if rst: rpp=-(rsp-rp)
                else: rpp=-(xp-rp)
                pts+=rpp; cnt+=1; re+=1
                legs.append((mid,re,rpp,'bear_poststop',pd.Timestamp(times[ri]),pd.Timestamp(times[rsi if rst else xi]),float(rp),float(rsp if rst else xp),'stop' if rst else 'natural'))
        week_entries[entry_wk]=week_entries.get(entry_wk,0)+1
        week_last[entry_wk]=pts
        masters.append((mid,d,dtype,pd.Timestamp(b.at[bi,'bar_start']),pd.Timestamp(b.at[entrybar,'bar_start']),pd.Timestamp(df.at[ei,'timestamp']),natural,pts,re,cnt,pd.Timestamp(df.at[ei,'timestamp']).year))
    m=pd.DataFrame(masters,columns=['master_id','d','type','signal_time','entry_bar','entry_time','natural','strategy_points','re','legs','year']); l=pd.DataFrame(legs,columns=['master_id','re','points','type','entry_time','exit_time','entry_price','exit_price','exit_reason'])
    mm=metrics(m.strategy_points);lm=metrics(l.points); stress=metrics(m.strategy_points.to_numpy()-2*m.legs.to_numpy())
    yr=m.groupby('year').strategy_points.sum();miss=m[(m.natural>=200)&(m.strategy_points<=0)]
    out=dict(net=mm[0],mdd=mm[1],mpf=mm[2],win=mm[3],masters=len(m),legs=len(l),re=int((l.re>0).sum()),legdd=lm[1],legpf=lm[2],stress2=stress[0],stress2dd=stress[1],stress2pf=stress[2],minyear=float(yr.min()),positive_years=int((yr>0).sum()),missed_large=len(miss),missed_nat=float(miss.natural.sum()),bull_delayed=int((m.type=='bull_delayed').sum()),bull_delayed_net=float(m.loc[m.type=='bull_delayed','strategy_points'].sum()),bull_resumption=int((m.type=='bull_resumption').sum()),bull_resumption_net=float(m.loc[m.type=='bull_resumption','strategy_points'].sum()),bull_breakout=int((m.type=='bull_breakout').sum()),bull_breakout_net=float(m.loc[m.type=='bull_breakout','strategy_points'].sum()),bull_weekly_monitor=int((m.type=='bull_weekly_monitor').sum()),bull_weekly_monitor_net=float(m.loc[m.type=='bull_weekly_monitor','strategy_points'].sum()),bull_reclaim=int((m.type=='bull_reclaim').sum()),bull_reclaim_net=float(m.loc[m.type=='bull_reclaim','strategy_points'].sum()))
    if frames:return out,l,m
    return out

def worker(args):
    vals=args;p=BASE.copy();p.update(vals);r=sim(p);r.update(vals);return r




# -----------------------------------------------------------------------------
# R3 parent-campaign rule family. These are strategy parameters only.
# -----------------------------------------------------------------------------
def r4_core_params(start, end):
    p=BASE.copy()
    p.update(dict(
        start_date=start,end_date=end,
        gap_exp=1.065,bear_body_max=.875,buffer=5.0,
        b1_rec=185.0,b1_wait=225,b1_gap=9.0,
        b2_rec=90.0,b2_wait=105,b2_gap=15.0,
        b3_rec=157.5,b3_wait=60,b3_gap=10.0,
        b4_rec=115.0,b4_wait=60,b4_gap=0.0,
        s1_rec=185.0,s1_wait=330,s1_gap=9.0,
        s2_rec=71.0,s2_wait=120,s2_gap=30.0,
        d_minbars=6,d_absgap=8.5,d_norm=.22,d_dist=22.5,d_ratio=1.0,
        dr_rec=15.0,dr_wait=240,dr_gap=20.0,dr_body=.50,dr_maxdist=55.0,
        ub_on=True,ub_minbars=5,ub_absgap=6,ub_norm=.26,ub_dist=12,ub_ratio=1.0,ub_maxre=0,
        ub_ret6min=2.0,ub_ret3max=-21,ub_maxdist=45,
        ubk_on=True,ubk_minbars=5,ubk_gap=6,ubk_norm=.47,ubk_ratio=1.04,ubk_dist=12,
        ubk_n=8,ubk_maxdist=55,ubk_ret6=27.5,ubk_body=.70,
        qim_on=True,qim_ret6=110,qim_dist=90,qim_body=.10,qim_norm=.020,
        qim_dyn_sl=False,qim_sl=10.0,
        weekly_gate_on=True,
        ubw_on=True,ubw_minbars=15,ubw_gap=6,ubw_norm=1.60,ubw_ratio=1.0,ubw_dist=12,
        ubw_maxdist=100,ubw_ret6=30,ubw_n=8,ubw_body=.70,
        ubr2_on=True,ubr2_minbars=8,ubr2_maxbars=999,ubr2_max_clock_min=2880,ubr2_arm_dist=-10,
        ubr2_norm=.05,ubr2_dist=5,ubr2_maxdist=25,ubr2_ret6=40,ubr2_body=.20,
        psu_on=False,psb_on=False,
        mb_on=True,mb_wait=60,mb_rec=10,mb_gap=6,mb_norm=.51,mb_ratio=1.0,
        mb_maxdist=40,mb_body=.60,mb_ret3=0,mb_ret6=0,
        normal_sl=10.0,db_sl=10.0,ub_sl=10.0,ubk_sl=10.0,ubw_sl=10.0,
        db_dyn_sl=False,ub_dyn_sl=False,ubk_dyn_sl=False,ubw_dyn_sl=False,
        core_stale_on=True,core_stale_minutes=90,core_stale_mfe=20.0,core_stale_max_pnl=0.0,
        core_stale_types=('normal','bear_delayed','bull_breakout','bull_quiet_momentum'),
    ))
    return p


def _core_frame(legs, masters):
    md=masters.set_index('master_id')['d'].to_dict()
    z=legs.copy()
    z['Direction']=z.master_id.map(lambda x:'LONG' if int(md[int(x)])==1 else 'SHORT')
    z['Entry_Time']=pd.to_datetime(z.entry_time); z['Exit_Time']=pd.to_datetime(z.exit_time)
    z['NIFTY_Entry']=z.entry_price.astype(float); z['NIFTY_Exit']=z.exit_price.astype(float)
    z['Points']=z.points.astype(float); z['Parent_Regime_ID']=z.master_id.astype(int)
    return z[['Parent_Regime_ID','Direction','Entry_Time','NIFTY_Entry','Exit_Time','NIFTY_Exit','Points']].sort_values('Entry_Time').reset_index(drop=True)


def _closed_metrics(points):
    a=np.asarray(points,float)
    eq=np.cumsum(a); pad=np.r_[0.,eq]; peak=np.maximum.accumulate(pad)
    gp=float(a[a>0].sum()); gl=float(-a[a<0].sum())
    return dict(net=float(a.sum()),pf=float(gp/gl) if gl else float('inf'),dd=float((peak-pad).max()),
                trades=int(len(a)),win_rate=float((a>0).mean()*100) if len(a) else 0.0)


def _monday(t):
    t=pd.Timestamp(t); return (t-pd.Timedelta(days=t.weekday())).normalize()


def run_swing_collector(df, core, *, swing_k=.75, breakout_bars=4, stop_points=10.0,
                        prior_abs_min=50.0, prior_atr_min=1.5, prior_er_max=.75,
                        prior_proof=20.0, prior_pull=30.0, prior_reclaim=10.0,
                        max_paid_per_swing=2, max_prior_loss_streak=7,
                        prior_loss_trigger=3, prior_cooldown_swings=1,
                        micro_abs_max=20.0, micro_atr_max=.25, micro_anchor_atr=2.0,
                        micro_proof=10.0, micro_pull=30.0, micro_reclaim=5.0):
    times=df.timestamp.to_numpy(dtype='datetime64[ns]'); op=df.open.to_numpy(float);hi=df.high.to_numpy(float);lo=df.low.to_numpy(float);N=len(df)
    # Core occupancy and realized loss state are generated from current-run trades only.
    diff=np.zeros(N+1,dtype=np.int16)
    for r in core.itertuples(index=False):
        a=np.searchsorted(times,np.datetime64(r.Entry_Time)); e=np.searchsorted(times,np.datetime64(r.Exit_Time))
        if a<N: diff[a]+=1
        if e<N: diff[e]-=1
    core_active=np.cumsum(diff[:-1])>0
    order=np.argsort(core.Exit_Time.to_numpy(dtype='datetime64[ns]'))
    core_exits=core.Exit_Time.to_numpy(dtype='datetime64[ns]')[order]; core_pnl=core.Points.to_numpy(float)[order]
    streaks=np.zeros(len(core_pnl),int);st=0
    for i,v in enumerate(core_pnl): st=st+1 if v<=0 else 0;streaks[i]=st
    def core_state(t):
        tt=np.datetime64(t);j=np.searchsorted(core_exits,tt,side='right')-1;ls=int(streaks[j]) if j>=0 else 0
        mon=np.datetime64(_monday(pd.Timestamp(t)));m=(core_exits>=mon)&(core_exits<=tt)
        return ls,float(core_pnl[m].sum())
    x=df.set_index('timestamp')
    b30=x.resample('30min',origin='start_day',offset='9h15min',label='left',closed='left').agg(high=('high','max'),low=('low','min'),close=('close','last')).dropna().reset_index()
    b30=b30[(b30.timestamp.dt.hour*60+b30.timestamp.dt.minute>=555)&(b30.timestamp.dt.hour*60+b30.timestamp.dt.minute<=915)].reset_index(drop=True)
    prev=b30.close.shift(1);tr=np.maximum(b30.high-b30.low,np.maximum((b30.high-prev).abs(),(b30.low-prev).abs()));b30['atr14']=tr.rolling(14,min_periods=14).mean()
    events=[];qualities=[];state=0;ext=None;leg_start=None;leg_i=None
    for ii,r in enumerate(b30.itertuples(index=False)):
        if not np.isfinite(r.atr14): continue
        c=float(r.close);th=swing_k*float(r.atr14);idx=np.searchsorted(times,np.datetime64(r.timestamp+pd.Timedelta(minutes=29)),side='right')
        if state==0: state=1;ext=c;leg_start=c;leg_i=ii;continue
        flip=False;nd=0;comp=0.0
        if state==1:
            ext=max(ext,c)
            if c<=ext-th: comp=max(0.,ext-leg_start);nd=-1;flip=True
        else:
            ext=min(ext,c)
            if c>=ext+th: comp=max(0.,leg_start-ext);nd=1;flip=True
        if flip:
            seg=b30.iloc[leg_i:ii+1].close.to_numpy(float);path=float(np.abs(np.diff(seg)).sum()) if len(seg)>1 else 0.;er=comp/path if path>0 else 0.
            if idx<N: events.append((idx,nd,pd.Timestamp(times[idx]),comp,float(r.atr14)));qualities.append(er)
            state=nd;leg_start=c;leg_i=ii;ext=c
    b5=x.resample('5min',origin='start_day',offset='9h15min',label='left',closed='left').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last')).dropna().reset_index()
    mm=b5.timestamp.dt.hour*60+b5.timestamp.dt.minute;b5=b5[(mm>=555)&(mm<=925)].reset_index(drop=True)
    b5['e5']=b5.close.ewm(span=5,adjust=False).mean();b5['e10']=b5.close.ewm(span=10,adjust=False).mean()
    b5['entry_idx']=[np.searchsorted(times,np.datetime64(t+pd.Timedelta(minutes=4)),side='right') for t in b5.timestamp]
    rh=b5.high.shift(1).rolling(breakout_bars,min_periods=breakout_bars).max();rl=b5.low.shift(1).rolling(breakout_bars,min_periods=breakout_bars).min()
    cand={1:b5.loc[(b5.e5>b5.e10)&(b5.close>rh)&(b5.entry_idx<N),'entry_idx'].to_numpy(int),-1:b5.loc[(b5.e5<b5.e10)&(b5.close<rl)&(b5.entry_idx<N),'entry_idx'].to_numpy(int)}
    def shadow(eidx,end,d,proof):
        ep=op[eidx];sl=ep-10 if d==1 else ep+10;tg=ep+proof if d==1 else ep-proof
        for i in range(eidx,end):
            if (lo[i]<=sl if d==1 else hi[i]>=sl): return None
            if (hi[i]>=tg if d==1 else lo[i]<=tg): return i
        return None
    def pull_reclaim(start,end,d,pull,reclaim):
        if start>=end-1:return None
        if d==1:
            fav=hi[start];armed=False;pb=np.inf
            for i in range(start+1,end):
                if not armed:
                    fav=max(fav,hi[i])
                    if lo[i]<=fav-pull:armed=True;pb=lo[i]
                else:
                    pb=min(pb,lo[i])
                    if hi[i]>=pb+reclaim:return i
        else:
            fav=lo[start];armed=False;pb=-np.inf
            for i in range(start+1,end):
                if not armed:
                    fav=min(fav,lo[i])
                    if hi[i]>=fav+pull:armed=True;pb=hi[i]
                else:
                    pb=max(pb,hi[i])
                    if lo[i]<=pb-reclaim:return i
        return None
    def micro(comp,atr): return comp<=micro_abs_max and (comp/atr if atr else 999)<=micro_atr_max
    rows=[];same_losses=0;blocked=None;prior_fail=0;cool_left=0
    for j,e in enumerate(events[:-1]):
        sidx,d,signal_time,completed,atr=e;end=events[j+1][0];ratio=completed/atr if atr else 0.;er=qualities[j]
        prior_ok=completed>=prior_abs_min and ratio>=prior_atr_min and er<=prior_er_max
        microchain=False
        if micro(completed,atr):
            if j-1>=0 and events[j-1][3]/events[j-1][4]>=micro_anchor_atr: microchain=True
            elif j-2>=0 and micro(events[j-1][3],events[j-1][4]) and events[j-2][3]/events[j-2][4]>=micro_anchor_atr: microchain=True
        if not(prior_ok or microchain):continue
        if same_losses>=2 and blocked is not None and signal_time!=blocked:same_losses=0;blocked=None
        route='PRIOR' if prior_ok else 'MICROCHAIN'
        if route=='PRIOR' and cool_left>0:
            cool_left-=1
            if cool_left==0:prior_fail=0
            continue
        proof,pull,reclaim=(prior_proof,prior_pull,prior_reclaim) if route=='PRIOR' else (micro_proof,micro_pull,micro_reclaim)
        arr=cand[d];a=np.searchsorted(arr,sidx);z=np.searchsorted(arr,end);arr=arr[a:z];cursor=sidx;paid=0;ci=0
        while ci<len(arr) and paid<max_paid_per_swing:
            while ci<len(arr) and arr[ci]<cursor:ci+=1
            if ci>=len(arr) or same_losses>=2:break
            ei=int(arr[ci]);ci+=1
            if core_active[ei]:continue
            pi=shadow(ei,end,d,proof)
            if pi is None:cursor=ei+1;continue
            ri0=pull_reclaim(pi,end,d,pull,reclaim)
            if ri0 is None:break
            ri=ri0+1
            if ri>=end:break
            if core_active[ri]:cursor=ri+1;continue
            entry_time=pd.Timestamp(times[ri]);ls,wk=core_state(entry_time)
            if route=='MICROCHAIN' and ls not in (2,3):cursor=ri+1;continue
            if route=='PRIOR' and ls>max_prior_loss_streak:cursor=ri+1;continue
            ep=float(op[ri]);paid+=1;sl=ep-stop_points if d==1 else ep+stop_points
            _hit=False; _xp=np.nan; _mfe=0.0; _protected=False
            for kk in range(ri,end):
                if d==1:
                    if op[kk]<=sl: _hit=True; xi=kk; _xp=float(op[kk]); break
                    if lo[kk]<=sl: _hit=True; xi=kk; _xp=float(sl); break
                    _mfe=max(_mfe,float(hi[kk]-ep))
                else:
                    if op[kk]>=sl: _hit=True; xi=kk; _xp=float(op[kk]); break
                    if hi[kk]>=sl: _hit=True; xi=kk; _xp=float(sl); break
                    _mfe=max(_mfe,float(ep-lo[kk]))
                # R3 ITM3 time-risk protection: after 20 completed market minutes,
                # a swing child that has already proven +40 MFE can no longer
                # return to its original -10 risk. A +5 protective stop becomes
                # active from the NEXT minute, avoiding same-bar hindsight.
                # A universal 30-minute forced exit was deliberately rejected:
                # the selective protection preserves the large parent runners.
                if (not _protected) and (kk-ri+1)>=20 and _mfe>=40.0:
                    sl=ep+5.0 if d==1 else ep-5.0; _protected=True
            if _hit:
                xp=float(_xp);pnl=float(d*(xp-ep));cursor=xi+1
                if pnl<=0:
                    same_losses+=1
                    if same_losses>=2:blocked=signal_time
                    if route=='PRIOR':
                        prior_fail+=1
                        if prior_fail>=prior_loss_trigger:cool_left=prior_cooldown_swings
                else:
                    same_losses=0;blocked=None
                    if route=='PRIOR':prior_fail=0
            else:
                xi=end;xp=float(op[xi]);pnl=d*(xp-ep);cursor=end;same_losses=0;blocked=None
                if route=='PRIOR':prior_fail=0
            rows.append((entry_time,pd.Timestamp(times[xi]),d,ep,xp,float(pnl),route,ls,wk))
            if not _hit:break
    return pd.DataFrame(rows,columns=['Entry_Time','Exit_Time','d','Entry','Exit','Points','Route','Core_Loss_Streak','Week_Realized'])



# -----------------------------------------------------------------------------
# R3 rule set: refined swing collector + sequential intraday week-repair rules.
# All constants below are strategy parameters, never historical outcomes.
# -----------------------------------------------------------------------------

def run_r3_swing(df, core):
    return run_swing_collector(
        df, core,
        swing_k=.75, breakout_bars=4, stop_points=10.0,
        prior_abs_min=50.0, prior_atr_min=1.5, prior_er_max=.75,
        prior_proof=20.0, prior_pull=25.0, prior_reclaim=5.0,
        max_paid_per_swing=3, max_prior_loss_streak=5,
        prior_loss_trigger=3, prior_cooldown_swings=1,
        micro_abs_max=15.0, micro_atr_max=.30, micro_anchor_atr=3.0,
        micro_proof=20.0, micro_pull=20.0, micro_reclaim=15.0,
    )


def _make_bars_with_indices(df, tf):
    x=df[['timestamp','open','high','low','close']].copy()
    minute=x.timestamp.dt.hour*60+x.timestamp.dt.minute
    x['day']=x.timestamp.dt.normalize()
    x['bucket']=((minute-555)//tf).astype(int)
    x['idx']=np.arange(len(x))
    return x.groupby(['day','bucket'],sort=False).agg(
        first_idx=('idx','first'), last_idx=('idx','last'),
        start=('timestamp','first'), end=('timestamp','last'),
        open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last')
    ).reset_index(drop=True)


def _tf_features(bars, pairs):
    close=bars.close.astype(float)
    prev=close.shift(1)
    tr=np.maximum((bars.high-bars.low).to_numpy(float),
                  np.maximum((bars.high-prev).abs().to_numpy(float),
                             (bars.low-prev).abs().to_numpy(float)))
    atr=pd.Series(tr).rolling(14,min_periods=14).mean()
    out=[]
    for fast,slow in pairs:
        ef=close.ewm(span=fast,adjust=False).mean()
        es=close.ewm(span=slow,adjust=False).mean()
        gap=(ef-es).abs()
        out.append((np.where(ef>es,1,-1).astype(np.int8),
                    (gap/atr).to_numpy(float),
                    (gap/gap.shift(1)).to_numpy(float),
                    ef.to_numpy(float),es.to_numpy(float)))
    return out


def _build_repair_features(df):
    b5=_make_bars_with_indices(df,5)
    b15=_make_bars_with_indices(df,15)
    b30=_make_bars_with_indices(df,30)
    p30=[(10,30),(19,29),(21,50)]
    p15=[(3,13),(5,21),(8,21),(13,34),(27,35)]
    p5=[(3,13),(5,10),(5,21),(8,21)]
    f30=_tf_features(b30,p30); f15=_tf_features(b15,p15); f5=_tf_features(b5,p5)
    l5=b5.last_idx.to_numpy(np.int64); l15=b15.last_idx.to_numpy(np.int64); l30=b30.last_idx.to_numpy(np.int64)
    i15=np.searchsorted(l15,l5,side='right')-1
    i30=np.searchsorted(l30,l5,side='right')-1
    P_DIR=np.zeros((len(p30),len(b5)),np.int8); P_GN=np.full((len(p30),len(b5)),np.nan); P_GR=np.full_like(P_GN,np.nan)
    C_DIR=np.zeros((len(p15),len(b5)),np.int8); C_GN=np.full((len(p15),len(b5)),np.nan); C_GR=np.full_like(C_GN,np.nan)
    M_DIR=np.zeros((len(p5),len(b5)),np.int8); M_GN=np.full((len(p5),len(b5)),np.nan); M_GR=np.full_like(M_GN,np.nan)
    M_EF=np.full((len(p5),len(b5)),np.nan); M_ES=np.full_like(M_EF,np.nan)
    for j,(d,gn,gr,ef,es) in enumerate(f30):
        ok=i30>=0; P_DIR[j,ok]=d[i30[ok]]; P_GN[j,ok]=gn[i30[ok]]; P_GR[j,ok]=gr[i30[ok]]
    for j,(d,gn,gr,ef,es) in enumerate(f15):
        ok=i15>=0; C_DIR[j,ok]=d[i15[ok]]; C_GN[j,ok]=gn[i15[ok]]; C_GR[j,ok]=gr[i15[ok]]
    for j,(d,gn,gr,ef,es) in enumerate(f5):
        M_DIR[j]=d; M_GN[j]=gn; M_GR[j]=gr; M_EF[j]=ef; M_ES[j]=es
    rng=(b5.high-b5.low).replace(0,np.nan)
    body=((b5.close-b5.open)/rng).to_numpy(float)
    path=b5.close.diff().abs().rolling(6).sum()
    er6=((b5.close-b5.close.shift(6)).abs()/path).to_numpy(float)
    ret3=(b5.close/b5.close.shift(3)-1).to_numpy(float)*10000.0
    return dict(
        b5=b5,p30=p30,p15=p15,p5=p5,
        P_DIR=P_DIR,P_GN=P_GN,P_GR=P_GR,C_DIR=C_DIR,C_GN=C_GN,C_GR=C_GR,
        M_DIR=M_DIR,M_GN=M_GN,M_GR=M_GR,M_EF=M_EF,M_ES=M_ES,
        body=body,er6=er6,ret3=ret3,
        entry=l5+1,b5last=l5,b5min=(b5.end.dt.hour*60+b5.end.dt.minute).to_numpy(np.int16)
    )


def _state_from_generated_trades(df, trades):
    """Build only live-known occupancy/realized-state arrays from current-run trades."""
    times=df.timestamp.to_numpy(dtype='datetime64[ns]'); n=len(df)
    dates=df.timestamp.dt.normalize()
    day_codes,unique_days=pd.factorize(dates); day_codes=day_codes.astype(np.int32)
    week_dates=dates-pd.to_timedelta(dates.dt.weekday,unit='D')
    week_codes,unique_weeks=pd.factorize(week_dates); week_codes=week_codes.astype(np.int32)
    day_last=np.zeros(len(unique_days),np.int64)
    for i,d in enumerate(day_codes): day_last[d]=i
    z=trades[['Entry_Time','Exit_Time','Points']].copy().sort_values(['Entry_Time','Exit_Time']).reset_index(drop=True)
    z['ei']=np.searchsorted(times,z.Entry_Time.to_numpy(dtype='datetime64[ns]'))
    z['xi']=np.searchsorted(times,z.Exit_Time.to_numpy(dtype='datetime64[ns]'))
    diff=np.zeros(n+1,np.int16); starts=np.zeros(n,np.bool_)
    events=[[] for _ in range(n)]
    for r in z.itertuples(index=False):
        a=min(int(r.ei),n-1); x=min(int(r.xi),n-1)
        diff[a]+=1; starts[a]=True
        if x+1<len(diff): diff[x+1]-=1
        events[x].append(float(r.Points))
    active=np.cumsum(diff[:-1])>0
    day_cum=np.zeros(n,float); week_cum=np.zeros(n,float)
    day_total=np.zeros(len(unique_days),float); week_total=np.zeros(len(unique_weeks),float)
    dc=-1;wc=-1;dp=0.;wp=0.
    for i in range(n):
        d=day_codes[i];w=week_codes[i]
        if d!=dc: dc=d;dp=0.
        if w!=wc: wc=w;wp=0.
        if events[i]:
            s=sum(events[i]); dp+=s;wp+=s;day_total[d]+=s;week_total[w]+=s
        day_cum[i]=dp;week_cum[i]=wp
    return dict(times=times,day_codes=day_codes,week_codes=week_codes,day_last=day_last,
                active=active,starts=starts,day_cum=day_cum,week_cum=week_cum,
                day_total=day_total,week_total=week_total)


def _run_week_repair(df, existing, feat, cfg, module_name):
    """Sequential, live-causal intraday reclaim module using generated current-run state only."""
    st=_state_from_generated_trades(df,existing)
    times=st['times']; day_codes=st['day_codes']; week_codes=st['week_codes']; day_last=st['day_last']
    active=st['active']; starts=st['starts']; dc=st['day_cum']; wc=st['week_cum']
    O=df.open.to_numpy(float);H=df.high.to_numpy(float);L=df.low.to_numpy(float);C=df.close.to_numpy(float)
    b5=feat['b5']; ENTRY=feat['entry']; B5LAST=feat['b5last']; B5MIN=feat['b5min']
    P_DIR=feat['P_DIR'];P_GN=feat['P_GN'];C_DIR=feat['C_DIR'];C_GN=feat['C_GN']
    M_DIR=feat['M_DIR'];M_GN=feat['M_GN'];M_GR=feat['M_GR'];M_EF=feat['M_EF']
    body=feat['body'];er6=feat['er6'];ret3=feat['ret3']
    pi=cfg['pi'];ci=cfg['ci'];mi=cfg['mi']
    repair_day=np.zeros(len(st['day_total']),float);repair_week=np.zeros(len(st['week_total']),float)
    rows=[];last_exit=-1;allow_i=0;current_day=-1;attempts=0
    for q in range(20,len(b5)-1):
        ei=int(ENTRY[q])
        if ei>=len(O) or day_codes[ei]!=day_codes[B5LAST[q]]: continue
        if ei<=last_exit or ei<allow_i or active[ei]: continue
        dday=int(day_codes[ei])
        if dday!=current_day: current_day=dday;attempts=0
        if attempts>=cfg['maxtr'] or B5MIN[q]<cfg['startm'] or B5MIN[q]>cfg['endm']: continue
        d=int(P_DIR[pi,q])
        if d==0 or int(C_DIR[ci,q])!=d or int(M_DIR[mi,q])!=d: continue
        if not np.isfinite(P_GN[pi,q]) or P_GN[pi,q]<cfg['pgn']: continue
        if not np.isfinite(C_GN[ci,q]) or C_GN[ci,q]<cfg['cgn']: continue
        if not np.isfinite(M_GN[mi,q]) or M_GN[mi,q]<cfg['mgn']: continue
        if not np.isfinite(M_GR[mi,q]) or M_GR[mi,q]<cfg['ratio']: continue
        if not np.isfinite(er6[q]) or er6[q]<cfg['er']: continue
        if d*ret3[q]<cfg['r3'] or d*body[q]<cfg['body']: continue
        # Completed 5m pullback to fast EMA, then close reclaims prior completed bar extreme.
        if q<1: continue
        ok=(b5.low.iloc[q-1]<=M_EF[mi,q-1] and b5.close.iloc[q]>b5.high.iloc[q-1]) if d==1 else \
           (b5.high.iloc[q-1]>=M_EF[mi,q-1] and b5.close.iloc[q]<b5.low.iloc[q-1])
        if not ok: continue
        dayp=dc[ei]+repair_day[dday]; ww=int(week_codes[ei]); weekp=wc[ei]+repair_week[ww]
        if weekp>cfg['weekgate'] or dayp>cfg['stopgreen']: continue
        endi=int(day_last[dday])
        ep=float(O[ei]); stop=ep-d*10.0; target=ep+d*cfg['target']; peak=ep;trough=ep
        xi=endi;xp=float(C[endi]);reason='EOD';mfe=0.0
        for i in range(ei,endi+1):
            # Higher-priority generated rule starts now: hand off at this minute open.
            if i>ei and starts[i]: xi=i;xp=float(O[i]);reason='HANDOFF';break
            if d==1:
                if O[i]<=stop: xi=i;xp=float(O[i]);reason='STOP_GAP' if O[i]<stop else 'STOP';break
                if L[i]<=stop: xi=i;xp=float(stop);reason='STOP';break
                if H[i]>=target: xi=i;xp=float(target);reason='TARGET';break
                peak=max(peak,float(H[i]));mfe=max(mfe,peak-ep)
                if cfg['be_arm']>0 and peak-ep>=cfg['be_arm']: stop=max(stop,ep+cfg['be_lock'])
                if cfg['trail_arm']>0 and peak-ep>=cfg['trail_arm']: stop=max(stop,peak-cfg['trail_dist'])
            else:
                if O[i]>=stop: xi=i;xp=float(O[i]);reason='STOP_GAP' if O[i]>stop else 'STOP';break
                if H[i]>=stop: xi=i;xp=float(stop);reason='STOP';break
                if L[i]<=target: xi=i;xp=float(target);reason='TARGET';break
                trough=min(trough,float(L[i]));mfe=max(mfe,ep-trough)
                if cfg['be_arm']>0 and ep-trough>=cfg['be_arm']: stop=min(stop,ep-cfg['be_lock'])
                if cfg['trail_arm']>0 and ep-trough>=cfg['trail_arm']: stop=min(stop,trough+cfg['trail_dist'])
        pnl=float(d*(xp-ep));repair_day[int(day_codes[xi])]+=pnl;repair_week[int(week_codes[xi])]+=pnl
        rows.append(dict(Module=module_name,Entry_Time=pd.Timestamp(times[ei]),Exit_Time=pd.Timestamp(times[xi]),
                         Direction='LONG' if d==1 else 'SHORT',Entry_Price=ep,Exit_Price=xp,Points=pnl,
                         Exit_Reason=reason,MFE=float(mfe),Day_PnL_At_Entry=float(dayp),Week_PnL_At_Entry=float(weekp),
                         Signal_Time=pd.Timestamp(b5.end.iloc[q]),Parent_30m=str(feat['p30'][pi]),
                         Child_15m=str(feat['p15'][ci]),Trigger_5m=str(feat['p5'][mi])))
        attempts+=1;last_exit=xi;allow_i=xi+cfg['cool']
    return pd.DataFrame(rows)


def _r3_repair_a_config():
    return dict(pi=1,ci=0,mi=0,pgn=.05,cgn=.60,mgn=.10,ratio=1.25,body=.40,er=.50,r3=5.0,
                target=70.0,be_arm=20.0,be_lock=0.0,trail_arm=70.0,trail_dist=20.0,
                maxtr=2,cool=90,startm=630,endm=900,weekgate=0.0,stopgreen=10.0)


def _r3_repair_b_config():
    # R3 weekly-repair plateau: balanced 30m/15m/5m alignment with a modest
    # negative-week gate. Parameters are rules only and use current-run state.
    return dict(pi=1,ci=0,mi=1,pgn=.05,cgn=.55,mgn=.05,ratio=1.10,body=.40,er=.40,r3=15.0,
                target=70.0,be_arm=50.0,be_lock=10.0,trail_arm=100.0,trail_dist=40.0,
                maxtr=2,cool=30,startm=615,endm=900,weekgate=-10.0,stopgreen=10.0)


def _combine_generated(core, swing, repair_a, repair_b):
    a=core[['Entry_Time','Exit_Time','Points']].copy();a['Module']='CORE'
    b=swing[['Entry_Time','Exit_Time','Points']].copy();b['Module']='SWING'
    frames=[a,b]
    if len(repair_a): frames.append(repair_a[['Entry_Time','Exit_Time','Points','Module']])
    if len(repair_b): frames.append(repair_b[['Entry_Time','Exit_Time','Points','Module']])
    return pd.concat(frames,ignore_index=True).sort_values(['Exit_Time','Entry_Time','Module']).reset_index(drop=True)


def _period_counts(df, realized):
    market_days=sorted(pd.to_datetime(df.timestamp).dt.date.unique())
    dsum=realized.assign(Date=pd.to_datetime(realized.Exit_Time).dt.date).groupby('Date').Points.sum()
    dv=np.array([float(dsum.get(d,0.0)) for d in market_days])
    monday=[d-pd.Timedelta(days=pd.Timestamp(d).weekday()) for d in pd.to_datetime(market_days)]
    all_weeks=sorted(set(pd.Timestamp(x).date() for x in monday))
    wsum=realized.assign(Week=pd.to_datetime(realized.Exit_Time).map(_monday)).groupby('Week').Points.sum()
    wv=np.array([float(wsum.get(pd.Timestamp(w),0.0)) for w in all_weeks])
    return dict(daily=dict(positive=int((dv>0).sum()),negative=int((dv<0).sum()),flat=int((dv==0).sum())),
                weekly=dict(positive=int((wv>0).sum()),negative=int((wv<0).sum()),flat=int((wv==0).sum())))



def _itm3_execution_segments(df, core, swing, *, profit_roll=70.0, max_sessions=2,
                             roll_hour=15, roll_minute=15):
    """Generate an option-execution roll plan from current-run rules only.

    This does NOT alter the parent NIFTY strategy P&L. It converts long-lived
    CORE/SWING parent exposure into short ITM3 execution children:
      * at each 15:15 session boundary, harvest/roll if child spot profit >= +70;
      * regardless of P&L, roll by the second trading session so the same
        weekly ITM3 contract is not intentionally carried through a long trend;
      * close and reopen at the same NIFTY reference, so child spot points
        telescope exactly back to the parent spot points.

    The parent trend remains alive after a roll. Actual strike/expiry selection
    belongs to the live options execution layer.
    """
    times=df.timestamp.to_numpy(dtype='datetime64[ns]')
    op=df.open.to_numpy(float)
    dates=pd.to_datetime(df.timestamp).dt.normalize()
    unique_days=list(pd.Index(dates.unique()).sort_values())
    day_pos={pd.Timestamp(d).date():i for i,d in enumerate(unique_days)}
    lookup={pd.Timestamp(ts):float(px) for ts,px in zip(df.timestamp,op)}

    parents=[]
    for r in core.itertuples(index=False):
        d=1 if str(r.Direction)=='LONG' else -1
        parents.append(dict(Parent_Module='CORE',Parent_ID=int(r.Parent_Regime_ID),Direction='LONG' if d==1 else 'SHORT',d=d,
                            Entry_Time=pd.Timestamp(r.Entry_Time),Exit_Time=pd.Timestamp(r.Exit_Time),
                            Entry_Price=float(r.NIFTY_Entry),Exit_Price=float(r.NIFTY_Exit)))
    for j,r in enumerate(swing.itertuples(index=False),1):
        d=int(r.d)
        parents.append(dict(Parent_Module='SWING',Parent_ID=j,Direction='LONG' if d==1 else 'SHORT',d=d,
                            Entry_Time=pd.Timestamp(r.Entry_Time),Exit_Time=pd.Timestamp(r.Exit_Time),
                            Entry_Price=float(r.Entry),Exit_Price=float(r.Exit)))

    out=[]
    for p in parents:
        e=p['Entry_Time']; x=p['Exit_Time']; d=p['d']
        if e.date() not in day_pos or x.date() not in day_pos or x<=e:
            continue
        child_start=e; child_px=p['Entry_Price']; child_session0=day_pos[e.date()]; seg=1
        end_day=day_pos[x.date()]
        k=child_session0
        while k<=end_day:
            bd=pd.Timestamp(unique_days[k])+pd.Timedelta(hours=roll_hour,minutes=roll_minute)
            if child_start < bd < x and bd in lookup:
                px=float(lookup[bd]); pnl=float(d*(px-child_px)); age=int(k-child_session0+1)
                reason=None
                if pnl>=profit_roll: reason='PROFIT_ROLL_70'
                elif age>=max_sessions: reason='AGE_ROLL_2D'
                if reason is not None:
                    out.append(dict(Parent_Module=p['Parent_Module'],Parent_ID=p['Parent_ID'],Segment_ID=seg,
                                    Direction=p['Direction'],Entry_Time=child_start,Exit_Time=bd,
                                    Entry_Price=float(child_px),Exit_Price=px,Spot_Points=pnl,
                                    Exit_Reason=reason,Trading_Sessions=age))
                    seg+=1;child_start=bd;child_px=px;child_session0=k
            k+=1
        age=max(1,end_day-child_session0+1)
        out.append(dict(Parent_Module=p['Parent_Module'],Parent_ID=p['Parent_ID'],Segment_ID=seg,
                        Direction=p['Direction'],Entry_Time=child_start,Exit_Time=x,
                        Entry_Price=float(child_px),Exit_Price=float(p['Exit_Price']),
                        Spot_Points=float(d*(p['Exit_Price']-child_px)),Exit_Reason='PARENT_EXIT',
                        Trading_Sessions=int(age)))
    return pd.DataFrame(out)


def _loss_duration_diagnostics(frames):
    rows=[]
    for z in frames:
        if z is None or len(z)==0: continue
        q=z.copy();q['Entry_Time']=pd.to_datetime(q.Entry_Time);q['Exit_Time']=pd.to_datetime(q.Exit_Time)
        q['Duration_Min']=(q.Exit_Time-q.Entry_Time).dt.total_seconds()/60.0
        rows.append(q[['Points','Duration_Min']])
    if not rows:return dict(losses=0,losses_over_30m=0,avg_loss_duration_min=0.0)
    q=pd.concat(rows,ignore_index=True);loss=q.Points<0
    return dict(losses=int(loss.sum()),losses_over_30m=int((loss&(q.Duration_Min>30)).sum()),
                avg_loss_duration_min=float(q.loc[loss,'Duration_Min'].mean()) if loss.any() else 0.0)

# -----------------------------------------------------------------------------
# Stable live-runner compatibility API. These aliases point to the exact R4
# rule functions above; they do not add or alter any strategy logic.
# -----------------------------------------------------------------------------
strategy_core_params = r4_core_params
run_strategy_swing = run_r3_swing
_strategy_repair_a_config = _r3_repair_a_config
_strategy_repair_b_config = _r3_repair_b_config


def run_engine(args):
    init(args.nifty)
    p=r4_core_params(args.start,args.end)
    _,legs,masters=sim(p,frames=True)
    core=_core_frame(legs,masters)
    sdf=DF[(DF.timestamp>=pd.Timestamp(args.start))&(DF.timestamp<pd.Timestamp(args.end))].copy().reset_index(drop=True)
    swing=run_r3_swing(sdf,core)
    feat=_build_repair_features(sdf)

    base=pd.concat([core[['Entry_Time','Exit_Time','Points']].assign(Module='CORE'),
                    swing[['Entry_Time','Exit_Time','Points']].assign(Module='SWING')],ignore_index=True)
    repair_a=_run_week_repair(sdf,base,feat,_r3_repair_a_config(),'REPAIR_A')
    with_a=pd.concat([base,repair_a[['Entry_Time','Exit_Time','Points','Module']]],ignore_index=True)
    repair_b=_run_week_repair(sdf,with_a,feat,_r3_repair_b_config(),'REPAIR_B')
    realized=_combine_generated(core,swing,repair_a,repair_b)
    itm3_segments=_itm3_execution_segments(sdf,core,swing,profit_roll=70.0,max_sessions=2)

    cm=_closed_metrics(core.sort_values(['Exit_Time','Entry_Time']).Points)
    sm=_closed_metrics(swing.sort_values(['Exit_Time','Entry_Time']).Points)
    am=_closed_metrics(repair_a.sort_values(['Exit_Time','Entry_Time']).Points) if len(repair_a) else _closed_metrics([])
    bm=_closed_metrics(repair_b.sort_values(['Exit_Time','Entry_Time']).Points) if len(repair_b) else _closed_metrics([])
    fm=_closed_metrics(realized.Points)
    years=[]
    yy=pd.to_datetime(realized.Exit_Time).dt.year
    for y in sorted(yy.unique()):
        z=realized[yy==y]
        years.append({'year':int(y),'points':float(z.Points.sum()),'trades':int(len(z))})
    counts=_period_counts(sdf,realized)
    risk_diag=_loss_duration_diagnostics([core[['Entry_Time','Exit_Time','Points']],
                                          swing[['Entry_Time','Exit_Time','Points']],
                                          repair_a[['Entry_Time','Exit_Time','Points']] if len(repair_a) else None,
                                          repair_b[['Entry_Time','Exit_Time','Points']] if len(repair_b) else None])
    roll_summary={'segments':int(len(itm3_segments)),'roll_boundaries':int((itm3_segments.Exit_Reason!='PARENT_EXIT').sum()) if len(itm3_segments) else 0,
                  'profit_rolls':int((itm3_segments.Exit_Reason=='PROFIT_ROLL_70').sum()) if len(itm3_segments) else 0,
                  'age_rolls':int((itm3_segments.Exit_Reason=='AGE_ROLL_2D').sum()) if len(itm3_segments) else 0,
                  'max_child_sessions':int(itm3_segments.Trading_Sessions.max()) if len(itm3_segments) else 0}
    summary={'core':cm,'swing':sm,'repair_a':am,'repair_b':bm,'combined':fm,
             'daily':counts['daily'],'weekly':counts['weekly'],'yearly':years,
             'risk_duration':risk_diag,'itm3_execution_segments':roll_summary}
    print(json.dumps(summary,indent=2))

    if args.out:
        from pathlib import Path
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        # Generated outputs only; nothing historical is embedded in this engine file.
        core.to_csv(out/'core_generated.csv',index=False)
        swing.to_csv(out/'swing_generated.csv',index=False)
        repair_a.to_csv(out/'repair_a_generated.csv',index=False)
        repair_b.to_csv(out/'repair_b_generated.csv',index=False)
        realized.to_csv(out/'all_generated.csv',index=False)
        itm3_segments.to_csv(out/'itm3_execution_segments_generated.csv',index=False)
        masters.to_csv(out/'core_masters_generated.csv',index=False)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))


def main():
    import argparse
    ap=argparse.ArgumentParser(description='Standalone rules-only NIFTY R4 engine (stable GitHub filename)')
    ap.add_argument('--nifty',required=True,help='raw NIFTY 1-minute CSV or ZIP')
    ap.add_argument('--start',required=True)
    ap.add_argument('--end',required=True,help='exclusive end')
    ap.add_argument('--out',default=None,help='optional directory for generated runtime reports')
    run_engine(ap.parse_args())


if __name__=='__main__':
    main()
