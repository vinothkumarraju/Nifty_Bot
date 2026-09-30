#!/usr/bin/env python3
"""
NIFTY RULES-ONLY TREND ENGINE R7
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

R6 accepted rule:
- Preserve the verified R5 A/B/C2145 architecture.
- R5-B may not initiate a new paid entry from 11:30 through 12:59 session time.
  Existing open trades and opposite-break exit signals remain unchanged.
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



# =============================================================================
# R5 COMPLEMENTARY WATCHDOG LAYERS (A -> B -> C2145)
# =============================================================================
# All rules below are computed sequentially from the supplied raw 1-minute OHLC.
# No historical trade rows, timestamps, P&L ledgers, or date-specific exceptions
# are embedded. These layers are allowed to act only when earlier-priority
# positions are flat. Completed 5m/15m bars decide; fills occur on the next
# regular-session 1m open.

R6_RULES = {
    'A': {
        'lookback_5m':4, 'day_move_min':40.0, 'efficiency_min':0.60,
        'prev_close_buffer':0.0, 'families':('rolling_breakout','ema_pullback_reclaim','prior_15m_breakout'),
        'protect_be_at_mfe':20.0, 'protect_25_at_mfe':45.0, 'protect_50_at_mfe':70.0,
        'shadow_proof_after_nonpositive':20.0, 'max_paid_per_day':4,
        'day_gate_below':10.0, 'week_gate_below':10.0, 'day_loss_floor':-20.0,
        'initial_stop':10.0,
    },
    'B': {
        'lookback_5m':3, 'day_move_min':0.0, 'efficiency_min':0.30,
        'prev_close_buffer':0.0, 'families':('rolling_breakout','ema_pullback_reclaim','opening_range_breakout','prior_15m_breakout'),
        'target':20.0, 'shadow_proof_after_nonpositive':20.0, 'max_paid_per_day':2,
        'day_gate_below':1.0, 'week_gate_below':1.0, 'day_loss_floor':-20.0,
        'initial_stop':10.0,
    },
    'C2145': {
        'ema_fast_15m':21, 'ema_slow_15m':45, 'cross_shadow_proof':25.0,
        'fresh_5m_breakout_lookback':4, 'protect_be_at_mfe':40.0,
        'protect_25_at_mfe':70.0, 'max_trading_days':2, 'max_rolls':1,
        'initial_stop':10.0,
    },
}


def _r5_session_features(sdf, base_trades):
    x=sdf[['timestamp','open','high','low','close']].copy().reset_index(drop=True)
    O=x.open.to_numpy(float); H=x.high.to_numpy(float); L=x.low.to_numpy(float); C=x.close.to_numpy(float)
    T=x.timestamp.to_numpy(dtype='datetime64[ns]'); N=len(x)
    D=pd.factorize(x.timestamp.dt.date)[0].astype(np.int32)
    years=x.timestamp.dt.year.to_numpy(np.int16)
    dates=np.array(x.timestamp.dt.date)
    starts=np.r_[0,np.flatnonzero(D[1:]!=D[:-1])+1].astype(np.int64)
    ends=np.r_[starts[1:]-1,N-1].astype(np.int64)
    nd=len(starts); udates=np.array([dates[s] for s in starts],object)
    prevclose=np.full(nd,np.nan)
    if nd>1: prevclose[1:]=C[ends[:-1]]
    wkdate=np.array([pd.Timestamp(d)-pd.Timedelta(days=pd.Timestamp(d).weekday()) for d in udates])
    uw=np.unique(wkdate); WDAY=np.searchsorted(uw,wkdate).astype(np.int32); W=WDAY[D]; nw=len(uw)

    # Earlier-priority (R4) occupancy and live realized day/week state.
    diff=np.zeros(N+1,np.int32); base_start=np.zeros(N,np.int8)
    add_day=np.zeros(N,float); add_week=np.zeros(N,float)
    day_real=np.zeros(nd,float); week_real=np.zeros(nw,float)
    bt=base_trades.copy()
    bt['Entry_Time']=pd.to_datetime(bt.Entry_Time); bt['Exit_Time']=pd.to_datetime(bt.Exit_Time)
    for r in bt.itertuples(index=False):
        s=int(np.searchsorted(T,np.datetime64(r.Entry_Time),'left'))
        e=int(np.searchsorted(T,np.datetime64(r.Exit_Time),'left'))
        if 0<=s<N:
            diff[s]+=1; base_start[s]=1
        if 0<=e<N:
            if e+1<=N: diff[e+1]-=1
            day_real[D[e]]+=float(r.Points); week_real[W[e]]+=float(r.Points)
            if e+1<N:
                if D[e+1]==D[e]: add_day[e+1]+=float(r.Points)
                if W[e+1]==W[e]: add_week[e+1]+=float(r.Points)
    occ=(np.cumsum(diff[:-1])>0).astype(np.int8)
    live_day=np.zeros(N,float); live_week=np.zeros(N,float)
    for s,e in zip(starts,ends): live_day[s:e+1]=np.cumsum(add_day[s:e+1])
    for wi in range(nw):
        ix=np.flatnonzero(W==wi)
        if len(ix): live_week[ix]=np.cumsum(add_week[ix])

    # Completed 5-minute bars.
    q=x.copy(); mm=q.timestamp.dt.hour*60+q.timestamp.dt.minute
    q['day']=q.timestamp.dt.normalize(); q['bucket']=((mm-555)//5).astype(np.int16); q['idx']=np.arange(N)
    b=q.groupby(['day','bucket'],sort=False).agg(last_idx=('idx','last'),bar_end=('timestamp','last'),open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),n=('idx','size')).reset_index(drop=True)
    li=b.last_idx.to_numpy(np.int64); bo=b.open.to_numpy(float); bh=b.high.to_numpy(float); bl=b.low.to_numpy(float); bc=b.close.to_numpy(float); bn=b.n.to_numpy(np.int16)
    NB=len(b); bD=pd.factorize(b.bar_end.dt.date)[0].astype(np.int32); bm=(b.bar_end.dt.hour*60+b.bar_end.dt.minute).to_numpy(np.int16)
    bs=np.r_[0,np.flatnonzero(bD[1:]!=bD[:-1])+1].astype(np.int64); be=np.r_[bs[1:]-1,NB-1].astype(np.int64)
    e5=ema(bc,5); e10=ema(bc,10)

    # Completed 15-minute bars and EMA5/10 alignment used by A/B.
    qq=x.copy(); m=qq.timestamp.dt.hour*60+qq.timestamp.dt.minute
    qq['day']=qq.timestamp.dt.normalize(); qq['bucket']=((m-555)//15).astype(np.int16); qq['idx']=np.arange(N)
    b15=qq.groupby(['day','bucket'],sort=False).agg(last_idx=('idx','last'),open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),n=('idx','size')).reset_index(drop=True)
    l15=b15.last_idx.to_numpy(np.int64); h15=b15.high.to_numpy(float); lo15=b15.low.to_numpy(float); c15=b15.close.to_numpy(float)
    e155=ema(c15,5); e1510=ema(c15,10)
    mp=np.searchsorted(l15,li,side='right')-1
    m15e5=np.where(mp>=0,e155[np.maximum(mp,0)],np.nan); m15e10=np.where(mp>=0,e1510[np.maximum(mp,0)],np.nan)
    m15h=np.where(mp>0,h15[np.maximum(mp-1,0)],np.nan); m15l=np.where(mp>0,lo15[np.maximum(mp-1,0)],np.nan)

    pc=np.full(NB,np.nan); dmove=np.zeros(NB); eff=np.zeros(NB); orh=np.full(NB,np.nan); orl=np.full(NB,np.nan)
    for d,(s,e) in enumerate(zip(bs,be)):
        pc[s:e+1]=prevclose[d]
        hi=np.maximum.accumulate(bh[s:e+1]); lo=np.minimum.accumulate(bl[s:e+1])
        dmove[s:e+1]=bc[s:e+1]-bo[s]
        rg=hi-lo; eff[s:e+1]=np.divide(np.abs(dmove[s:e+1]),rg,out=np.zeros_like(rg),where=rg>0)
        oe=min(e,s+5); orh[s:e+1]=np.max(bh[s:oe+1]); orl[s:e+1]=np.min(bl[s:oe+1])
    rng=bh-bl; body=np.divide(bc-bo,rng,out=np.zeros_like(rng),where=rng>0)
    prevc=np.r_[np.nan,bc[:-1]]; prevh=np.r_[np.nan,bh[:-1]]; prevl=np.r_[np.nan,bl[:-1]]
    preve10=np.r_[np.nan,e10[:-1]]
    roll={}
    for n in (3,4,5):
        rh=np.full(NB,np.nan); rl=np.full(NB,np.nan)
        for s,e in zip(bs,be):
            for i in range(s+n,e+1):
                rh[i]=np.max(bh[i-n:i]); rl[i]=np.min(bl[i-n:i])
        roll[n]=(rh,rl)
    nxt=li+1; same=(nxt<N)
    valid_n=np.flatnonzero(same)
    same[valid_n]=D[nxt[valid_n]]==bD[valid_n]

    return dict(x=x,O=O,H=H,L=L,C=C,T=T,N=N,D=D,W=W,years=years,starts=starts,ends=ends,nd=nd,nw=nw,
                occ=occ,start=base_start,live_day=live_day,live_week=live_week,day_real=day_real,week_real=week_real,
                b=b,li=li,bo=bo,bh=bh,bl=bl,bc=bc,bn=bn,NB=NB,bD=bD,bm=bm,bs=bs,be=be,e5=e5,e10=e10,
                b15=b15,l15=l15,h15=h15,lo15=lo15,c15=c15,m15e5=m15e5,m15e10=m15e10,m15h=m15h,m15l=m15l,
                pc=pc,dmove=dmove,eff=eff,orh=orh,orl=orl,body=body,prevc=prevc,prevh=prevh,prevl=prevl,preve10=preve10,
                roll=roll,nxt=nxt,same=same)


def _r5_build_signal(F,n,move,min_eff,buf,fammask,bodymin=.05,start_min=580,end_min=915):
    rh,rl=F['roll'][n]; bc=F['bc']; pc=F['pc']; dmove=F['dmove']; eff=F['eff']; e5=F['e5']; e10=F['e10']
    valid=(F['bn']>=5)&(F['bm']>=start_min)&(F['bm']<=end_min)&np.isfinite(pc)&F['same']
    trendL=(bc>pc+buf)&(dmove>=move)&(eff>=min_eff)&(e5>e10)&(F['m15e5']>F['m15e10'])
    trendS=(bc<pc-buf)&(dmove<=-move)&(eff>=min_eff)&(e5<e10)&(F['m15e5']<F['m15e10'])
    Lsig=np.zeros(F['NB'],bool); Ssig=np.zeros(F['NB'],bool)
    if fammask&1:
        Lsig|=(bc>rh)&(F['prevc']<=rh); Ssig|=(bc<rl)&(F['prevc']>=rl)
    if fammask&2:
        pbL=(F['prevl']<=F['preve10'])&(bc>F['prevh'])&(bc>e5)&(F['body']>=bodymin)
        pbS=(F['prevh']>=F['preve10'])&(bc<F['prevl'])&(bc<e5)&(F['body']<=-bodymin)
        Lsig|=pbL; Ssig|=pbS
    if fammask&4:
        Lsig|=(bc>F['orh'])&(F['prevc']<=F['orh']); Ssig|=(bc<F['orl'])&(F['prevc']>=F['orl'])
    if fammask&8:
        Lsig|=(bc>F['m15h'])&(F['prevc']<=F['m15h']); Ssig|=(bc<F['m15l'])&(F['prevc']>=F['m15l'])
    L=valid&trendL&Lsig; S=valid&trendS&Ssig
    eL=np.zeros(F['N'],np.int8); eS=np.zeros(F['N'],np.int8); rL=np.zeros(F['N'],np.int8); rS=np.zeros(F['N'],np.int8)
    eL[F['nxt'][L]]=1; eS[F['nxt'][S]]=1
    rL[F['nxt'][valid&(bc>rh)]]=1; rS[F['nxt'][valid&(bc<rl)]]=1
    return eL,eS,rL,rS


def _r5_sim_breakout(F,eL,eS,rL,rS,occ,start_arr,baseDay,baseWeek,arm0,arm25,arm50,proof,maxpaid,target,day_gate,week_gate,max_day_loss,layer):
    O,H,L,C,D,W=F['O'],F['H'],F['L'],F['C'],F['D'],F['W']; ends=F['ends']; N=F['N']
    rows=[]; i=0; cd=-1; cw=-1; paid=0; shadow=False; wdD=0.; wdW=0.
    while i<N:
        d=int(D[i]); w=int(W[i])
        if w!=cw: cw=w; wdW=0.
        if d!=cd: cd=d; paid=0; shadow=False; wdD=0.
        if paid>=maxpaid or occ[i]>0 or (eL[i]==0 and eS[i]==0): i+=1; continue
        if baseDay[i]+wdD>=day_gate or baseWeek[i]+wdW>=week_gate or baseDay[i]+wdD<=-max_day_loss: i+=1; continue
        di=1 if eL[i] else -1
        signal_i=i
        if shadow:
            ve=float(O[i]); j=i; ok=False
            while j<=ends[d]:
                if occ[j]>0 or start_arr[j]>0: break
                if (di==1 and rS[j]) or (di==-1 and rL[j]): break
                if di==1:
                    if O[j]<=ve-10 or L[j]<=ve-10: break
                    if H[j]>=ve+proof:
                        if j+1<=ends[d] and occ[j+1]==0 and start_arr[j+1]==0: i=j+1; ok=True
                        break
                else:
                    if O[j]>=ve+10 or H[j]>=ve+10: break
                    if L[j]<=ve-proof:
                        if j+1<=ends[d] and occ[j+1]==0 and start_arr[j+1]==0: i=j+1; ok=True
                        break
                j+=1
            if not ok:
                i=max(i+1,j+1); continue
            if baseDay[i]+wdD>=day_gate or baseWeek[i]+wdW>=week_gate: continue
        if i>=N or D[i]!=d or occ[i]>0: i+=1; continue
        ep=float(O[i]); ei=i; mfe=0.; paid+=1; j=i; xp=ep; reason='DAY_END'
        while j<=ends[d]:
            if j>i and start_arr[j]>0: xp=float(O[j]); reason='R4_OR_PRIOR_LAYER_HANDOFF'; break
            if j>i and ((di==1 and rS[j]) or (di==-1 and rL[j])): xp=float(O[j]); reason='OPPOSITE_5M_BREAK'; break
            floor=-10.
            if mfe>=arm0: floor=0.
            if mfe>=arm25: floor=25.
            if mfe>=arm50: floor=50.
            sl=ep+di*floor
            if di==1:
                if O[j]<=sl: xp=float(O[j]); reason='STOP_GAP'; break
                if L[j]<=sl: xp=float(sl); reason='STOP'; break
                if target<999 and H[j]>=ep+target: xp=float(ep+target); mfe=max(mfe,target); reason='TARGET'; break
                mfe=max(mfe,float(H[j]-ep))
            else:
                if O[j]>=sl: xp=float(O[j]); reason='STOP_GAP'; break
                if H[j]>=sl: xp=float(sl); reason='STOP'; break
                if target<999 and L[j]<=ep-target: xp=float(ep-target); mfe=max(mfe,target); reason='TARGET'; break
                mfe=max(mfe,float(ep-L[j]))
            if j==ends[d]: xp=float(C[j]); reason='DAY_END'; break
            j+=1
        p=float(di*(xp-ep)); wdD+=p; wdW+=p
        rows.append(dict(Module=layer,Direction='LONG' if di==1 else 'SHORT',Signal_Time=pd.Timestamp(F['T'][signal_i]),
                         Entry_Time=pd.Timestamp(F['T'][ei]),Entry_Price=ep,Exit_Time=pd.Timestamp(F['T'][j]),Exit_Price=xp,
                         Points=p,Exit_Reason=reason,MFE=float(mfe),Initial_Stop=10.0))
        shadow=p<=0; i=j+1
    return pd.DataFrame(rows)


def _r5_add_layer(F,trades,occ,start_arr,live_day,live_week,day_real,week_real):
    occo=occ.copy(); sto=start_arr.copy(); addD=np.zeros(F['N']); addW=np.zeros(F['N']); trD=np.zeros(F['nd']); trW=np.zeros(F['nw'])
    if trades is not None and len(trades):
        for r in trades.itertuples(index=False):
            s=int(np.searchsorted(F['T'],np.datetime64(r.Entry_Time),'left')); e=int(np.searchsorted(F['T'],np.datetime64(r.Exit_Time),'left')); pp=float(r.Points)
            if not (0<=s<F['N'] and 0<=e<F['N']): continue
            occo[s:e+1]=1; sto[s]=1; trD[F['D'][e]]+=pp; trW[F['W'][e]]+=pp
            if e+1<F['N']:
                if F['D'][e+1]==F['D'][e]: addD[e+1]+=pp
                if F['W'][e+1]==F['W'][e]: addW[e+1]+=pp
    dl=live_day.copy(); wl=live_week.copy()
    for ss,ee in zip(F['starts'],F['ends']): dl[ss:ee+1]+=np.cumsum(addD[ss:ee+1])
    for wi in range(F['nw']):
        ix=np.flatnonzero(F['W']==wi)
        if len(ix): wl[ix]+=np.cumsum(addW[ix])
    return occo,sto,dl,wl,day_real+trD,week_real+trW


def _r5_c2145_arrays(F):
    c15=F['c15']; fast=ema(c15,21); slow=ema(c15,45); gap=fast-slow; sign=np.sign(gap); prev=np.r_[0,sign[:-1]]
    trend=np.zeros(F['N'],np.int8); cross=np.zeros(F['N'],np.int8)
    l15=F['l15']
    for k in range(len(l15)):
        ii=int(l15[k]+1)
        if ii>=F['N']: continue
        jj=F['N'] if k+1==len(l15) else int(l15[k+1]+1)
        d=1 if gap[k]>0 else -1; trend[ii:jj]=d
        if sign[k]>0 and prev[k]<=0: cross[ii]=1
        elif sign[k]<0 and prev[k]>=0: cross[ii]=-1
    # Fresh completed 5m EMA5/10 breakout of previous four completed 5m bars.
    rh4,rl4=F['roll'][4]; br=np.zeros(F['N'],np.int8)
    valid=(F['bn']>=5)&F['same']&np.isfinite(rh4)&np.isfinite(rl4)
    blg=valid&(F['e5']>F['e10'])&(F['bc']>rh4)&(F['prevc']<=rh4)
    bsg=valid&(F['e5']<F['e10'])&(F['bc']<rl4)&(F['prevc']>=rl4)
    br[F['nxt'][blg]]=1; br[F['nxt'][bsg]]=-1
    return trend,cross,br


def _r5_sim_c2145(F,occ,start_arr,trend,cross,br5):
    O,H,L,C,D=F['O'],F['H'],F['L'],F['C'],F['D']; ends=F['ends']; N=F['N']
    rows=[]; i=0; pos=False; di=0; ep=0.; ei=0; mfe=0.; mend=0; pending=0; rolls=0; entry_reason=''
    while i<N:
        dday=int(D[i])
        if cross[i]!=0:
            pending=int(cross[i]); rolls=0
        if not pos:
            if occ[i]>0: i+=1; continue
            td=int(trend[i]); enter=False; entry_i=i; signal_i=i
            if cross[i]!=0:
                di=int(cross[i]); pending=0
                # Shadow the raw 15m EMA21/45 crossover until +25 proves before -10.
                ve=float(O[i]); j=i; ok=False
                while j<=ends[dday]:
                    if occ[j]>0 or start_arr[j]>0: break
                    if trend[j]!=0 and trend[j]!=di: break
                    if di==1:
                        if O[j]<=ve-10 or L[j]<=ve-10: break
                        if H[j]>=ve+25:
                            if j+1<=ends[dday] and occ[j+1]==0 and start_arr[j+1]==0 and trend[j+1]==di:
                                entry_i=j+1; ok=True
                            break
                    else:
                        if O[j]>=ve+10 or H[j]>=ve+10: break
                        if L[j]<=ve-25:
                            if j+1<=ends[dday] and occ[j+1]==0 and start_arr[j+1]==0 and trend[j+1]==di:
                                entry_i=j+1; ok=True
                            break
                    j+=1
                if not ok:
                    i=max(i+1,j+1); continue
                i=entry_i; dday=int(D[i]); enter=True; entry_reason='CROSS_PROOF'
            elif pending!=0 and td==pending and br5[i]==td:
                di=td; enter=True; pending=0; entry_reason='PENDING_5M_BREAKOUT'
            elif rolls>0 and rolls<=1 and td==di and br5[i]==td:
                enter=True; entry_reason='TWO_DAY_ROLL_REENTRY'
            if not enter: i+=1; continue
            pos=True; ep=float(O[i]); ei=i; mfe=0.; targetd=int(D[i])+1
            if targetd>=len(ends): targetd=len(ends)-1
            mend=int(ends[targetd])
        exit_now=False; xp=ep; reason=''
        if i>ei and start_arr[i]>0: xp=float(O[i]); reason='R4_OR_PRIOR_LAYER_HANDOFF'; exit_now=True
        elif i>ei and trend[i]!=0 and trend[i]!=di: xp=float(O[i]); reason='EMA21_45_REVERSAL'; exit_now=True
        else:
            floor=-10.
            if mfe>=40: floor=0.
            if mfe>=70: floor=25.
            sl=ep+di*floor
            if di==1:
                if O[i]<=sl: xp=float(O[i]); reason='STOP_GAP'; exit_now=True
                elif L[i]<=sl: xp=float(sl); reason='STOP'; exit_now=True
                else: mfe=max(mfe,float(H[i]-ep))
            else:
                if O[i]>=sl: xp=float(O[i]); reason='STOP_GAP'; exit_now=True
                elif H[i]>=sl: xp=float(sl); reason='STOP'; exit_now=True
                else: mfe=max(mfe,float(ep-L[i]))
            if (not exit_now) and i>=mend: xp=float(C[i]); reason='TWO_DAY_EXIT'; exit_now=True
        if not exit_now: i+=1; continue
        pp=float(di*(xp-ep))
        rows.append(dict(Module='C2145',Direction='LONG' if di==1 else 'SHORT',Signal_Time=pd.Timestamp(F['T'][signal_i]),
                         Entry_Time=pd.Timestamp(F['T'][ei]),Entry_Price=ep,Exit_Time=pd.Timestamp(F['T'][i]),Exit_Price=xp,
                         Points=pp,Exit_Reason=reason,MFE=float(mfe),Initial_Stop=10.0,Entry_Reason=entry_reason))
        pos=False
        if reason=='TWO_DAY_EXIT' and trend[i]==di: rolls+=1
        elif reason in ('R4_OR_PRIOR_LAYER_HANDOFF','EMA21_45_REVERSAL'): rolls=0
        i+=1
    return pd.DataFrame(rows)


def _r5_combined_metrics(realized):
    q=realized.sort_values(['Exit_Time','Entry_Time']).reset_index(drop=True)
    p=q.Points.to_numpy(float); gp=float(p[p>0].sum()); gl=float(-p[p<0].sum())
    eq=np.cumsum(p); pk=np.maximum.accumulate(np.r_[0.,eq]); dd=float(np.max(pk[1:]-eq)) if len(p) else 0.0
    return {'net':float(p.sum()),'pf':float(gp/gl if gl else np.inf),'dd':dd,'trades':int(len(p)),'win_rate':float((p>0).mean()*100 if len(p) else 0.0)}


# =============================================================================
# R6 ACCEPTED FILTER: B MIDDAY NO-ENTRY WINDOW
# =============================================================================
# This is a live-known session-time filter only. It blocks NEW R5-B entries
# from 11:30 through 12:59. It does not alter already-open trades or the
# opposite-break arrays used for exits. No date, historical trade, or P&L
# lookup is used.
R6_B_NO_ENTRY_START_MIN = 11*60 + 30
R6_B_NO_ENTRY_END_MIN = 13*60

def _r6_filter_b_midday_entries(F, sig):
    eL,eS,rL,rS=[x.copy() for x in sig]
    minute=F['x'].timestamp.dt.hour.to_numpy()*60 + F['x'].timestamp.dt.minute.to_numpy()
    blocked=(minute>=R6_B_NO_ENTRY_START_MIN)&(minute<R6_B_NO_ENTRY_END_MIN)
    eL[blocked]=0
    eS[blocked]=0
    return eL,eS,rL,rS


# =============================================================================
# R7 ACCEPTED SHORT-CHILD LAYERS
# =============================================================================
# R7 extends the verified R6 engine with live-causal, runtime-generated child
# trades. No historical trade row, timestamp, date exception, or precomputed
# P&L is embedded. All layer state (day/week P&L, loss streak, occupancy) is
# rebuilt from trades generated earlier in the SAME run.
R7_RULES = {
    'R7_1_IMPULSE': {
        'tf':'15m->5m','expansion_min':1.60,'body_min':0.70,'confirm_minutes':45,
        'loss_streak_min':2,'week_min':-10.0,'week_max':0.0,'day_max':10.0,
        'exclude_start_min':630,'exclude_end_min':840,'initial_stop':10.0,
        'be_mfe':30.0,'lock_mfe':70.0,'lock_points':25.0,'trail_arm':120.0,'trail_dist':50.0,
        'max_trading_days':2,'max_per_day':1,
    },
    'R7_2_QUIET_PARENT': {
        'compression_ratio':0.62,'body_min':0.40,'week_min':-20.0,'week_max':0.0,
        'loss_streak_min':1,'shadow_proof':10.0,'shadow_fail':10.0,'proof_window_minutes':45,
        'initial_stop':10.0,'be_mfe':30.0,'lock_mfe':70.0,'lock_points':25.0,
        'trail_arm':120.0,'trail_dist':50.0,'max_trading_days':2,'max_per_day':1,
    },
    'R7_3_STRONG_MICRO': {
        'expansion_min':2.10,'body_min':0.55,'week_min':-5.0,'week_max':0.0,
        'loss_streak_min':2,'loss_streak_max':3,'target':35.0,'initial_stop':10.0,
        'be_mfe':15.0,'lock_mfe':30.0,'lock_points':5.0,'max_per_day':1,
        'exclude_1':[765,840],'exclude_2':[870,930],
    },
    'R7_4_STRONG_CHILD': {
        'expansion_min':2.05,'body_min':0.60,'breakout_extension_min':0.05,
        'week_min':-10.0,'week_max':0.0,'loss_streak_min':0,'loss_streak_max':2,
        'target':40.0,'initial_stop':10.0,'be_mfe':15.0,'lock_mfe':30.0,'lock_points':5.0,
        'max_per_day':1,'exclude_1':[765,840],'exclude_2':[870,930],
    },
    'R7_5_INTERMEDIATE': {
        'expansion_min':1.90,'expansion_max':2.05,'body_min':0.60,'breakout_extension_min':0.10,
        'prior3_compression_max':1.50,'week_min':-10.0,'week_max':0.0,
        'loss_streak_min':0,'loss_streak_max':2,'target':40.0,'initial_stop':10.0,
        'be_mfe':15.0,'lock_mfe':30.0,'lock_points':5.0,'max_per_day':1,
        'exclude_1':[765,840],'exclude_2':[870,930],
    },
    'R7_6_SECOND_CHILD': {
        'expansion_min':1.90,'body_min':0.55,'ema_gap_norm_min':0.10,'breakout_extension_min':0.10,
        'min_minutes_since_prior_1p9_signal':20,'week_min':-10.0,'week_max':0.0,
        'loss_streak_min':1,'loss_streak_max':3,'target':25.0,'initial_stop':10.0,
        'be_mfe':15.0,'lock_mfe':30.0,'lock_points':5.0,'max_per_day':2,
        'exclude_1':[765,840],'exclude_2':[870,930],
    },
    'R7_7_MORNING_PROOF_LONG': {
        'direction':'LONG','body_min':0.60,'week_min':-10.0,'week_max':0.0,
        'loss_streak_min':1,'loss_streak_max':3,'shadow_proof':18.0,'proof_window_minutes':45,
        'signal_end_min':750,'target':20.0,'initial_stop':10.0,'be_mfe':15.0,
        'lock_mfe':25.0,'lock_points':5.0,'max_per_day':1,
    },
    'R7_8_LATE_CHILD': {
        'include_start_min':840,'include_end_min':870,'expansion_min':1.60,'body_min':0.55,
        'ema_gap_norm_min':0.15,'breakout_extension_min':0.03,'week_min':-10.0,'week_max':0.0,
        'loss_streak_min':0,'loss_streak_max':2,'target':20.0,'initial_stop':10.0,
        'be_mfe':15.0,'lock_mfe':30.0,'lock_points':5.0,'max_per_day':1,
    },
}


def _r7_monday(t):
    t=pd.Timestamp(t); return (t-pd.Timedelta(days=t.weekday())).normalize()


def _r7_detail_from_r6(core,swing,repair_a,repair_b,A,B5,C15):
    parts=[]
    if len(core): parts.append(pd.DataFrame({'Module':'CORE','Direction':core.Direction,'Entry_Time':core.Entry_Time,'Exit_Time':core.Exit_Time,'Points':core.Points}))
    if len(swing): parts.append(pd.DataFrame({'Module':'SWING','Direction':np.where(swing.d.to_numpy(float)>0,'LONG','SHORT'),'Entry_Time':swing.Entry_Time,'Exit_Time':swing.Exit_Time,'Points':swing.Points}))
    for z,name in [(repair_a,'REPAIR_A'),(repair_b,'REPAIR_B'),(A,'R5_A'),(B5,'R5_B'),(C15,'C2145')]:
        if z is None or not len(z): continue
        direction=z.Direction if 'Direction' in z.columns else np.where(z.d.to_numpy(float)>0,'LONG','SHORT')
        module=z.Module if 'Module' in z.columns else name
        parts.append(pd.DataFrame({'Module':module,'Direction':direction,'Entry_Time':z.Entry_Time,'Exit_Time':z.Exit_Time,'Points':z.Points}))
    q=pd.concat(parts,ignore_index=True)
    q.Entry_Time=pd.to_datetime(q.Entry_Time); q.Exit_Time=pd.to_datetime(q.Exit_Time)
    return q.sort_values(['Exit_Time','Entry_Time','Module']).reset_index(drop=True)


def _r7_simple_layer(name,T,oe,ox,op,od):
    if len(op)==0:
        return pd.DataFrame(columns=['Module','Direction','Entry_Time','Exit_Time','Points'])
    return pd.DataFrame({'Module':name,'Direction':np.where(np.asarray(od)>0,'LONG','SHORT'),
                         'Entry_Time':pd.to_datetime(T[np.asarray(oe,dtype=int)]),
                         'Exit_Time':pd.to_datetime(T[np.asarray(ox,dtype=int)]),
                         'Points':np.asarray(op,float)})


def _r7_metrics_detail(q):
    q=q.sort_values(['Exit_Time','Entry_Time']).reset_index(drop=True)
    p=q.Points.to_numpy(float); gp=float(p[p>0].sum()); gl=float(-p[p<0].sum())
    pad=np.r_[0.,np.cumsum(p)]; pk=np.maximum.accumulate(pad)
    return {'net':float(p.sum()),'pf':float(gp/gl if gl else np.inf),'dd':float((pk-pad).max()),
            'trades':int(len(p)),'win_rate':float((p>0).mean()*100 if len(p) else 0.)}


def _r7_period_counts(df,q):
    days=sorted(pd.to_datetime(df.timestamp).dt.date.unique())
    weeks=sorted(set(_r7_monday(v) for v in days))
    qq=q.copy(); qq['D2']=pd.to_datetime(qq.Exit_Time).dt.date; qq['W2']=pd.to_datetime(qq.Exit_Time).map(_r7_monday)
    ds=qq.groupby('D2').Points.sum(); ws=qq.groupby('W2').Points.sum()
    dv=np.array([float(ds.get(x,0.)) for x in days]); wv=np.array([float(ws.get(x,0.)) for x in weeks])
    return {'daily':{'positive':int((dv>0).sum()),'negative':int((dv<0).sum()),'flat':int((dv==0).sum())},
            'weekly':{'green':int((wv>0).sum()),'red':int((wv<0).sum()),'flat':int((wv==0).sum())}}


def _r7_ids(df):
    dates=pd.to_datetime(df.timestamp).dt.normalize(); D,ud=pd.factorize(dates); D=D.astype(np.int32)
    wdate=dates-pd.to_timedelta(dates.dt.weekday,unit='D'); W,uw=pd.factorize(wdate); W=W.astype(np.int32)
    N=len(df); starts=np.r_[0,np.flatnonzero(D[1:]!=D[:-1])+1].astype(int); ends=np.r_[starts[1:]-1,N-1].astype(int)
    day_end=np.empty(N,int)
    for s,e in zip(starts,ends): day_end[s:e+1]=e
    return D,ud,W,uw,starts,ends,day_end


def _r7_state(df,base,next_minute_realization=False):
    T=pd.to_datetime(df.timestamp).to_numpy('datetime64[ns]'); N=len(df); D,ud,W,uw,starts,ends,day_end=_r7_ids(df)
    ent=np.searchsorted(T,pd.to_datetime(base.Entry_Time).to_numpy('datetime64[ns]'),'left').astype(np.int64)
    ext=np.searchsorted(T,pd.to_datetime(base.Exit_Time).to_numpy('datetime64[ns]'),'left').astype(np.int64)
    diff=np.zeros(N+1,np.int32); bst=np.zeros(N,np.int8)
    for a,e in zip(ent,ext):
        if a<N: diff[a]+=1; bst[a]=1
        if e<N and e+1<=N: diff[e+1]-=1
    occ=np.cumsum(diff[:-1])>0
    exitp=np.zeros(N,float)
    for e,p in zip(ext,base.Points.to_numpy(float)):
        if e<N:
            ix=e+1 if next_minute_realization else e
            if ix<N: exitp[ix]+=float(p)
    live_day=np.zeros(N,float); live_week=np.zeros(N,float)
    for s,e in zip(starts,ends): live_day[s:e+1]=np.cumsum(exitp[s:e+1])
    wstarts=np.r_[0,np.flatnonzero(W[1:]!=W[:-1])+1].astype(int); wends=np.r_[wstarts[1:]-1,N-1].astype(int)
    for s,e in zip(wstarts,wends): live_week[s:e+1]=np.cumsum(exitp[s:e+1])
    # dynamic loss streak, available at exit minute or next minute according to mode
    mark=np.full(N,np.nan); st=0
    order=np.argsort(ext,kind='stable')
    for j in order:
        e=int(ext[j]); p=float(base.Points.iloc[j]); st=st+1 if p<=0 else 0
        ix=e+1 if next_minute_realization else e
        if 0<=ix<N: mark[ix]=st
    lsarr=pd.Series(mark).ffill().fillna(0).to_numpy(np.int16)
    nextbst=np.full(N,N,np.int64); nb=N
    for i in range(N-1,-1,-1):
        if bst[i]: nb=i
        nextbst[i]=nb
    oo=np.argsort(ext,kind='stable'); bext=ext[oo]; bp=base.Points.to_numpy(float)[oo]
    return {'T':T,'D':D,'ud':ud,'W':W,'uw':uw,'starts':starts,'ends':ends,'day_end':day_end,
            'ent':ent,'ext':ext,'occ':occ,'bst':bst,'live_day':live_day,'live_week':live_week,
            'lsarr':lsarr,'nextbst':nextbst,'bext':bext,'bp':bp}


def _r7_bars(df,tf):
    x=df.copy(); m=x.timestamp.dt.hour*60+x.timestamp.dt.minute; x['day']=x.timestamp.dt.normalize(); x['bucket']=((m-555)//tf).astype(int); x['idx']=np.arange(len(x))
    return x.groupby(['day','bucket'],sort=False).agg(first_idx=('idx','first'),last_idx=('idx','last'),start=('timestamp','first'),end=('timestamp','last'),open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),cnt=('idx','size')).reset_index(drop=True)


def _r7_common_context(df):
    T=pd.to_datetime(df.timestamp).to_numpy('datetime64[ns]'); O=df.open.to_numpy(float); H=df.high.to_numpy(float); L=df.low.to_numpy(float); C=df.close.to_numpy(float); N=len(df)
    D,ud,W,uw,starts,ends,day_end=_r7_ids(df)
    b5=_r7_bars(df,5); b15=_r7_bars(df,15); b30=_r7_bars(df,30)
    for b in (b5,b15,b30):
        b['range']=b.high-b.low; b['bodyr']=(b.close-b.open)/b['range'].replace(0,np.nan)
    for b in (b5,b15):
        b['e5']=b.close.ewm(span=5,adjust=False).mean(); b['e10']=b.close.ewm(span=10,adjust=False).mean()
    b30['e19']=b30.close.ewm(span=19,adjust=False).mean(); b30['e29']=b30.close.ewm(span=29,adjust=False).mean()
    b5['rh4']=b5.high.shift(1).rolling(4,min_periods=4).max(); b5['rl4']=b5.low.shift(1).rolling(4,min_periods=4).min(); b5['med12']=b5['range'].shift(1).rolling(12,min_periods=8).median(); b5['avg3']=b5['range'].shift(1).rolling(3,min_periods=3).mean()
    b15['rh4']=b15.high.shift(1).rolling(4,min_periods=4).max(); b15['rl4']=b15.low.shift(1).rolling(4,min_periods=4).min(); b15['med8']=b15['range'].shift(1).rolling(8,min_periods=6).median(); b15['exp']=b15['range']/b15.med8
    b15['prevclose']=b15.close.shift(1); tr=np.maximum((b15.high-b15.low).to_numpy(float),np.maximum((b15.high-b15.prevclose).abs().to_numpy(float),(b15.low-b15.prevclose).abs().to_numpy(float)))
    b15['atr14']=pd.Series(tr).rolling(14,min_periods=14).mean(); b15['gap']=abs(b15.e5-b15.e10); b15['gapnorm']=b15.gap/b15.atr14; b15['gapratio']=b15.gap/b15.gap.shift(1)
    li5=b5.last_idx.to_numpy(np.int64); li15=b15.last_idx.to_numpy(np.int64); li30=b30.last_idx.to_numpy(np.int64)
    # completed 15m state mapped to minutes
    map15_m=np.searchsorted(li15,np.arange(N),side='right')-1; d15_m=np.zeros(N,np.int8); ok=map15_m>=0; jj=np.maximum(map15_m,0); e155=b15.e5.to_numpy(float); e1510=b15.e10.to_numpy(float); d15_m[ok]=np.where(e155[jj[ok]]>e1510[jj[ok]],1,-1)
    # micro candidates
    nxt5=li5+1; valid5=nxt5<N; vi=np.flatnonzero(valid5); valid5[vi]=D[nxt5[vi]]==D[li5[vi]]
    map15_5=np.searchsorted(li15,li5,side='right')-1; gn15=b15.gapnorm.to_numpy(float); gr15=b15.gapratio.to_numpy(float)
    d15_5=np.zeros(len(b5),np.int8); g15_5=np.full(len(b5),np.nan); gr15_5=np.full(len(b5),np.nan); ok=map15_5>=0; jj=np.maximum(map15_5,0)
    d15_5[ok]=np.where(e155[jj[ok]]>e1510[jj[ok]],1,-1); g15_5[ok]=gn15[jj[ok]]; gr15_5[ok]=gr15[jj[ok]]
    rng=b5['range'].to_numpy(float); med=b5.med12.to_numpy(float); body=np.abs(b5.bodyr.to_numpy(float)); body_signed=b5.bodyr.to_numpy(float); e5=b5.e5.to_numpy(float); e10=b5.e10.to_numpy(float); rh=b5.rh4.to_numpy(float); rl=b5.rl4.to_numpy(float); bc=b5.close.to_numpy(float)
    rows=[]; qrows=[]
    for q in range(len(b5)):
        if not valid5[q] or int(b5.cnt.iloc[q])<5 or not np.isfinite(med[q]) or med[q]<=0: continue
        ex=float(rng[q]/med[q]); d=0; br=0.
        if d15_5[q]==1 and e5[q]>e10[q] and np.isfinite(rh[q]) and bc[q]>rh[q] and body_signed[q]>0:
            d=1; br=float((bc[q]-rh[q])/max(rng[q],1e-9))
        elif d15_5[q]==-1 and e5[q]<e10[q] and np.isfinite(rl[q]) and bc[q]<rl[q] and body_signed[q]<0:
            d=-1; br=float((rl[q]-bc[q])/max(rng[q],1e-9))
        if d:
            ei=int(nxt5[q]); mn=pd.Timestamp(T[ei]).hour*60+pd.Timestamp(T[ei]).minute
            precomp=float(np.mean(rng[q-3:q])/med[q]) if q>=3 and np.isfinite(med[q]) and med[q]>0 else 999.
            rows.append((ei,d,ex,float(body[q]),float(g15_5[q]) if np.isfinite(g15_5[q]) else -1.,float(gr15_5[q]) if np.isfinite(gr15_5[q]) else -1.,br,mn,precomp,q))
    # first19 / since19 across candidate sequence, matching research harness
    cnt_by_day={}; last_by_day={}; out=[]
    for row in rows:
        ei=row[0]; dd=int(D[ei]); cnt=cnt_by_day.get(dd,0); last=last_by_day.get(dd,None); since=(ei-last) if last is not None else 9999
        out.append(row+(cnt,since))
        if row[2]>=1.90:
            cnt_by_day[dd]=cnt+1; last_by_day[dd]=ei
    cand=np.array(out,float) if out else np.empty((0,12),float)
    return {'T':T,'O':O,'H':H,'L':L,'C':C,'N':N,'D':D,'ud':ud,'W':W,'uw':uw,'starts':starts,'ends':ends,'day_end':day_end,
            'b5':b5,'b15':b15,'b30':b30,'li5':li5,'li15':li15,'li30':li30,'d15_m':d15_m,'cand':cand}


def _r7_dynamic_streak_at(base_ext,base_pts,entry_i,bj,streak,lastproc):
    while bj<len(base_ext) and base_ext[bj]<=entry_i:
        if base_ext[bj]>lastproc:
            streak=streak+1 if base_pts[bj]<=0 else 0
        bj+=1
    return bj,streak,max(lastproc,entry_i)


def _r7_sim_micro(ctx,base,cfg,name):
    st=_r7_state(pd.DataFrame({'timestamp':pd.to_datetime(ctx['T'])}),base) if False else None
    # build state using the original OHLC frame stored externally by caller through ctx['df']
    state=_r7_state(ctx['df'],base,False)
    T,O,H,L,C,D,W,day_end=ctx['T'],ctx['O'],ctx['H'],ctx['L'],ctx['C'],ctx['D'],ctx['W'],ctx['day_end']; cand=ctx['cand']; d15_m=ctx['d15_m']; N=ctx['N']
    occ,bst,ld,lw,bext,bp,nextbst=state['occ'],state['bst'],state['live_day'],state['live_week'],state['bext'],state['bp'],state['nextbst']
    addW=np.zeros(len(state['uw'])); addD=np.zeros(len(state['ud'])); used=np.zeros(len(state['ud']),np.int16)
    rows=[]; busy=-1; bj=0; streak=0; lastproc=-1
    for r in cand:
        ei=int(r[0]); d=int(r[1]); sexp,sbody,sgn,sgr,sbr,mn,precomp,first19,since19=float(r[2]),float(r[3]),float(r[4]),float(r[5]),float(r[6]),int(r[7]),float(r[8]),int(r[10]),int(r[11])
        bj,streak,lastproc=_r7_dynamic_streak_at(bext,bp,ei,bj,streak,lastproc)
        if ei<=busy or occ[ei]: continue
        if sexp<cfg.get('expansion_min',-1e9) or sexp>=cfg.get('expansion_max',1e9): continue
        if sbody<cfg.get('body_min',0.) or sgn<cfg.get('ema_gap_norm_min',-1e9) or sgr<cfg.get('ema_gap_ratio_min',-1e9) or sbr<cfg.get('breakout_extension_min',-1e9): continue
        if precomp>cfg.get('prior3_compression_max',1e9) or first19>cfg.get('first19_max',999999) or since19<cfg.get('min_since_1p9',0): continue
        mode=cfg.get('direction_mode',0)
        if mode==1 and d!=1: continue
        if mode==-1 and d!=-1: continue
        if cfg.get('include_start_min') is not None and not (cfg['include_start_min']<=mn<cfg['include_end_min']): continue
        for key in ('exclude_1','exclude_2'):
            w=cfg.get(key)
            if w is not None and w[0]<=mn<w[1]: break
        else:
            w=None
        if w is not None: continue
        dd=int(D[ei]); ww=int(W[ei])
        if used[dd]>=cfg.get('max_per_day',1): continue
        wk=float(lw[ei]+addW[ww]); dy=float(ld[ei]+addD[dd])
        if wk<cfg.get('week_min',-1e9) or wk>cfg.get('week_max',1e9) or streak<cfg.get('loss_streak_min',0) or streak>cfg.get('loss_streak_max',999999) or dy>=cfg.get('day_max',20.): continue
        zend=int(day_end[ei]); nb=int(nextbst[ei]);
        if nb>ei and nb<zend: zend=nb
        ep=float(O[ei]); mfe=0.; xp=float(C[zend]); xi=zend; reason='DAY_END'
        for j in range(ei,zend+1):
            if j>ei and bst[j]: xp=float(O[j]); xi=j; reason='BASE_HANDOFF'; break
            if j>ei and d15_m[j]!=0 and d15_m[j]!=d: xp=float(O[j]); xi=j; reason='15M_REV'; break
            floor=-float(cfg.get('initial_stop',10.))
            if mfe>=cfg.get('be_mfe',15.): floor=0.
            if mfe>=cfg.get('lock_mfe',30.): floor=float(cfg.get('lock_points',5.))
            sl=ep+d*floor; target=float(cfg.get('target',1e9))
            if d==1:
                if O[j]<=sl: xp=float(O[j]); xi=j; reason='STOP_GAP'; break
                if L[j]<=sl: xp=float(sl); xi=j; reason='STOP'; break
                if H[j]>=ep+target: xp=ep+target; xi=j; mfe=max(mfe,target); reason='TARGET'; break
                mfe=max(mfe,float(H[j]-ep))
            else:
                if O[j]>=sl: xp=float(O[j]); xi=j; reason='STOP_GAP'; break
                if H[j]>=sl: xp=float(sl); xi=j; reason='STOP'; break
                if L[j]<=ep-target: xp=ep-target; xi=j; mfe=max(mfe,target); reason='TARGET'; break
                mfe=max(mfe,float(ep-L[j]))
        pp=float(d*(xp-ep)); addW[ww]+=pp; addD[dd]+=pp; used[dd]+=1; busy=xi
        bj,streak,lastproc=_r7_dynamic_streak_at(bext,bp,xi,bj,streak,lastproc)
        streak=streak+1 if pp<=0 else 0; lastproc=max(lastproc,xi)
        rows.append({'Module':name,'Direction':'LONG' if d==1 else 'SHORT','Entry_Time':pd.Timestamp(T[ei]),'Exit_Time':pd.Timestamp(T[xi]),'Points':pp,'MFE':float(mfe),'Exit_Reason':reason})
    return pd.DataFrame(rows)


def _r7_layer1(df,base,ctx):
    state=_r7_state(df,base,False); T,O,H,L,C,N=ctx['T'],ctx['O'],ctx['H'],ctx['L'],ctx['C'],ctx['N']; D,W=ctx['D'],ctx['W']; day_end=ctx['day_end']; b5,b15=ctx['b5'],ctx['b15']; d15_m=ctx['d15_m']
    # fresh completed 5m breakout candidate by direction
    cand={1:[], -1:[]}; li5=b5.last_idx.to_numpy(int)
    e5=b5.e5.to_numpy(float);e10=b5.e10.to_numpy(float);rh=b5.rh4.to_numpy(float);rl=b5.rl4.to_numpy(float);bc=b5.close.to_numpy(float)
    for q,r in b5.iterrows():
        if int(r.cnt)<5: continue
        ei=int(r.last_idx)+1
        if ei>=N or D[ei]!=D[int(r.last_idx)]: continue
        if e5[q]>e10[q] and np.isfinite(rh[q]) and bc[q]>rh[q]: cand[1].append((ei,q))
        if e5[q]<e10[q] and np.isfinite(rl[q]) and bc[q]<rl[q]: cand[-1].append((ei,q))
    for d in cand: cand[d]=np.array(cand[d],dtype=int) if cand[d] else np.empty((0,2),int)
    sig=[]
    for q,r in b15.iterrows():
        if int(r.cnt)<15 or not np.isfinite(r.exp) or float(r.exp)<1.6 or abs(float(r.bodyr))<.70: continue
        d=0
        if r.e5>r.e10 and r.bodyr>0 and np.isfinite(r.rh4) and r.close>r.rh4: d=1
        elif r.e5<r.e10 and r.bodyr<0 and np.isfinite(r.rl4) and r.close<r.rl4: d=-1
        if not d: continue
        s=int(r.last_idx)+1
        if s>=N: continue
        arr=cand[d]; a=np.searchsorted(arr[:,0],s); z=np.searchsorted(arr[:,0],min(s+45,N-1),side='right')
        for row in arr[a:z]:
            ei=int(row[0])
            if D[ei]!=D[s]: break
            sig.append((ei,d)); break
    sig=sorted(sig)
    occ,bst,ld,lw,lsarr,nextbst=state['occ'],state['bst'],state['live_day'],state['live_week'],state['lsarr'],state['nextbst']
    # two-session map
    unique_days=list(pd.Index(pd.to_datetime(df.timestamp).dt.normalize().unique()).sort_values()); dpos={pd.Timestamp(v):i for i,v in enumerate(unique_days)}; dnorm=pd.to_datetime(df.timestamp).dt.normalize().to_numpy()
    def two_end(ei):
        p=dpos[pd.Timestamp(df.timestamp.iloc[ei]).normalize()]; td=unique_days[min(p+1,len(unique_days)-1)].to_datetime64(); ix=np.flatnonzero(dnorm==td); return int(ix[-1])
    addW=np.zeros(len(state['uw']));addD=np.zeros(len(state['ud']));used=np.zeros(len(state['ud']),np.int8);busy=-1;rows=[]
    for ei,d in sig:
        if ei<=busy or occ[ei] or lsarr[ei]<2: continue
        mn=pd.Timestamp(T[ei]).hour*60+pd.Timestamp(T[ei]).minute
        if 630<=mn<840: continue
        dd=int(D[ei]);ww=int(W[ei]);wk=float(lw[ei]+addW[ww]);dy=float(ld[ei]+addD[dd])
        if wk<-10 or wk>0 or dy>=10 or used[dd]>=1: continue
        zend=two_end(ei); nb=int(nextbst[ei]);
        if nb>ei and nb<zend: zend=nb
        ep=float(O[ei]);mfe=0.;xp=float(C[zend]);xi=zend;reason='TIME2D'
        for j in range(ei,zend+1):
            if j>ei and bst[j]:xp=float(O[j]);xi=j;reason='BASE_HANDOFF';break
            if j>ei and d15_m[j]!=0 and d15_m[j]!=d:xp=float(O[j]);xi=j;reason='15M_REV';break
            floor=-10.
            if mfe>=30:floor=0.
            if mfe>=70:floor=25.
            if mfe>=120:floor=max(floor,mfe-50.)
            sl=ep+d*floor
            if d==1:
                if O[j]<=sl:xp=float(O[j]);xi=j;reason='STOP_GAP';break
                if L[j]<=sl:xp=float(sl);xi=j;reason='STOP';break
                mfe=max(mfe,float(H[j]-ep))
            else:
                if O[j]>=sl:xp=float(O[j]);xi=j;reason='STOP_GAP';break
                if H[j]>=sl:xp=float(sl);xi=j;reason='STOP';break
                mfe=max(mfe,float(ep-L[j]))
        pp=float(d*(xp-ep));addW[ww]+=pp;addD[dd]+=pp;used[dd]+=1;busy=xi
        rows.append({'Module':'R7_1_IMPULSE','Direction':'LONG' if d==1 else 'SHORT','Entry_Time':pd.Timestamp(T[ei]),'Exit_Time':pd.Timestamp(T[xi]),'Points':pp,'MFE':mfe,'Exit_Reason':reason})
    return pd.DataFrame(rows)


def _r7_layer2(df,base,ctx):
    # exact R7-2 uses next-minute realization of prior layer state.
    state=_r7_state(df,base,True); T,O,H,L,C,N=ctx['T'],ctx['O'],ctx['H'],ctx['L'],ctx['C'],ctx['N'];D,W=ctx['D'],ctx['W']; day_end=ctx['day_end'];b5,b15,b30=ctx['b5'],ctx['b15'],ctx['b30'];d15_m=ctx['d15_m']
    li5=b5.last_idx.to_numpy(int);li15=b15.last_idx.to_numpy(int);li30=b30.last_idx.to_numpy(int);nxt=li5+1;valid=nxt<N;ii=np.flatnonzero(valid);valid[ii]=D[nxt[ii]]==D[li5[ii]]
    mp15=np.searchsorted(li15,li5,side='right')-1;mp30=np.searchsorted(li30,li5,side='right')-1
    D15=np.zeros(len(b5),np.int8);D30=np.zeros(len(b5),np.int8);ok=mp15>=0;D15[ok]=np.where(b15.e5.to_numpy()[mp15[ok]]>b15.e10.to_numpy()[mp15[ok]],1,-1);ok=mp30>=0;D30[ok]=np.where(b30.e19.to_numpy()[mp30[ok]]>b30.e29.to_numpy()[mp30[ok]],1,-1)
    qc=b5.avg3.to_numpy(float)<=.62*b5.med12.to_numpy(float);body=b5.bodyr.to_numpy(float);bc=b5.close.to_numpy(float);e5=b5.e5.to_numpy(float);e10=b5.e10.to_numpy(float);rh=b5.rh4.to_numpy(float);rl=b5.rl4.to_numpy(float);mins=(b5.end.dt.hour*60+b5.end.dt.minute).to_numpy(int)
    up=(bc>rh)&(e5>e10)&np.isfinite(rh);dn=(bc<rl)&(e5<e10)&np.isfinite(rl)
    Lm=valid&(D30==1)&(D15==1)&qc&up&(body>=.40)&(mins>=585)&(mins<=885); Sm=valid&(D30==-1)&(D15==-1)&qc&dn&(body<=-.40)&(mins>=585)&(mins<=885)
    sig=sorted([(int(nxt[i]),1) for i in np.flatnonzero(Lm)]+[(int(nxt[i]),-1) for i in np.flatnonzero(Sm)])
    occ,bst,ld,lw,lsarr,nextbst=state['occ'],state['bst'],state['live_day'],state['live_week'],state['lsarr'],state['nextbst']
    unique_days=list(pd.Index(pd.to_datetime(df.timestamp).dt.normalize().unique()).sort_values());dpos={pd.Timestamp(v):i for i,v in enumerate(unique_days)};dnorm=pd.to_datetime(df.timestamp).dt.normalize().to_numpy()
    def two_end(ei):
        p=dpos[pd.Timestamp(df.timestamp.iloc[ei]).normalize()];td=unique_days[min(p+1,len(unique_days)-1)].to_datetime64();ix=np.flatnonzero(dnorm==td);return int(ix[-1])
    addW=np.zeros(len(state['uw']));addD=np.zeros(len(state['ud']));used=np.zeros(len(state['ud']),np.int8);busy=-1;rows=[]
    for si,d in sig:
        if si<=busy or occ[si]:continue
        dd=int(D[si]);ww=int(W[si]);wk=float(lw[si]+addW[ww]);dy=float(ld[si]+addD[dd])
        if wk<-20 or wk>0 or dy>=10 or lsarr[si]<1 or used[dd]>=1:continue
        ref=float(O[si]);lim=min(int(day_end[si]),si+45);hit=-1
        for j in range(si,lim+1):
            if occ[j] or (j>si and bst[j]):break
            if d==1:
                if O[j]<=ref-10 or L[j]<=ref-10:break
                if H[j]>=ref+10:hit=j;break
            else:
                if O[j]>=ref+10 or H[j]>=ref+10:break
                if L[j]<=ref-10:hit=j;break
        if hit<0 or hit+1>day_end[si]:continue
        ei=hit+1
        if occ[ei]:continue
        zend=two_end(ei);nb=int(nextbst[ei]);
        if nb>ei and nb<zend:zend=nb
        ep=float(O[ei]);mfe=0.;xp=float(C[zend]);xi=zend;reason='TIME2D'
        for j in range(ei,zend+1):
            if j>ei and bst[j]:xp=float(O[j]);xi=j;reason='BASE_HANDOFF';break
            if j>ei and d15_m[j]!=0 and d15_m[j]!=d:xp=float(O[j]);xi=j;reason='15M_REV';break
            floor=-10.
            if mfe>=30:floor=0.
            if mfe>=70:floor=25.
            if mfe>=120:floor=max(floor,mfe-50.)
            sl=ep+d*floor
            if d==1:
                if O[j]<=sl:xp=float(O[j]);xi=j;reason='STOP_GAP';break
                if L[j]<=sl:xp=float(sl);xi=j;reason='STOP';break
                mfe=max(mfe,float(H[j]-ep))
            else:
                if O[j]>=sl:xp=float(O[j]);xi=j;reason='STOP_GAP';break
                if H[j]>=sl:xp=float(sl);xi=j;reason='STOP';break
                mfe=max(mfe,float(ep-L[j]))
        pp=float(d*(xp-ep));addW[ww]+=pp;addD[dd]+=pp;used[dd]+=1;busy=xi
        rows.append({'Module':'R7_2_QUIET_PARENT','Direction':'LONG' if d==1 else 'SHORT','Entry_Time':pd.Timestamp(T[ei]),'Exit_Time':pd.Timestamp(T[xi]),'Points':pp,'MFE':mfe,'Exit_Reason':reason})
    return pd.DataFrame(rows)



try:
    from numba import njit as _r7_njit
except Exception:
    def _r7_njit(*args, **kwargs):
        def deco(fn): return fn
        return deco

@_r7_njit(cache=False)
def _r7_morning_sim_fast(O,H,L,C,D,W,occ,bst,day_end,d15_m,ld,lw,bext,bp,nextbst,ca):
    nw=int(W[-1])+1;nd=int(D[-1])+1;addW=np.zeros(nw);addD=np.zeros(nd);used=np.zeros(nd,np.int16)
    oe=np.empty(len(ca),np.int64);ox=np.empty(len(ca),np.int64);op=np.empty(len(ca));od=np.empty(len(ca),np.int8);n=0
    bj=0;streak=0;lastproc=-1;busy=-1
    for k in range(len(ca)):
        ei0=int(ca[k,0]);d=int(ca[k,2]);bucket=int(ca[k,3]);mn=555+(bucket+1)*5
        while bj<len(bext) and bext[bj]<=ei0:
            if bext[bj]>lastproc:
                if bp[bj]<=0:streak+=1
                else:streak=0
            bj+=1
        if ei0>lastproc:lastproc=ei0
        if ei0<=busy or occ[ei0]!=0 or mn>=750 or ca[k,13]<0.60:continue
        if d!=1:continue
        lvl=ca[k,7]
        if np.isnan(lvl) or ca[k,4]<=lvl:continue
        dd=D[ei0];ww=W[ei0]
        if used[dd]>=1:continue
        wk=lw[ei0]+addW[ww];dy=ld[ei0]+addD[dd]
        if wk<-10. or wk>0. or streak<1 or streak>3 or dy>=20.:continue
        ref=O[ei0];pe=-1;lim=day_end[ei0]
        if ei0+45<lim:lim=ei0+45
        for j in range(ei0,lim+1):
            if j>ei0 and bst[j]!=0:break
            if L[j]<=ref-10.:break
            if H[j]>=ref+18.:pe=j;break
        if pe<0 or pe+1>=len(O) or D[pe+1]!=dd:continue
        ei=pe+1
        if occ[ei]!=0:continue
        zend=day_end[ei];nb=nextbst[ei]
        if nb>ei and nb<zend:zend=nb
        ep=O[ei];mfe=0.;xp=C[zend];xi=zend
        for j in range(ei,zend+1):
            if j>ei and bst[j]!=0:xp=O[j];xi=j;break
            if j>ei and d15_m[j]!=0 and d15_m[j]!=1:xp=O[j];xi=j;break
            floor=-10.
            if mfe>=15.:floor=0.
            if mfe>=25.:floor=5.
            sl=ep+floor
            if O[j]<=sl:xp=O[j];xi=j;break
            if L[j]<=sl:xp=sl;xi=j;break
            if H[j]>=ep+20.:xp=ep+20.;xi=j;break
            if H[j]-ep>mfe:mfe=H[j]-ep
        pp=xp-ep;addW[ww]+=pp;addD[dd]+=pp;used[dd]+=1;busy=xi
        while bj<len(bext) and bext[bj]<=xi:
            if bext[bj]>lastproc:
                if bp[bj]<=0:streak+=1
                else:streak=0
            bj+=1
        if pp<=0:streak+=1
        else:streak=0
        if xi>lastproc:lastproc=xi
        oe[n]=ei;ox[n]=xi;op[n]=pp;od[n]=1;n+=1
    return oe[:n],ox[:n],op[:n],od[:n]

def _r7_morning_candidates(ctx):
    T,O,H,L,C,N,D=ctx['T'],ctx['O'],ctx['H'],ctx['L'],ctx['C'],ctx['N'],ctx['D'];d15_m=ctx['d15_m']
    x=pd.DataFrame({'timestamp':pd.to_datetime(T),'open':O,'high':H,'low':L,'close':C,'idx':np.arange(N)})
    x['date']=x.timestamp.dt.normalize(); mm=x.timestamp.dt.hour*60+x.timestamp.dt.minute; x['bucket']=((mm-555)//5).astype(int)
    b5=x.groupby(['date','bucket'],sort=False).agg(last_idx=('idx','last'),open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),cnt=('idx','size')).reset_index()
    b5['ema5']=b5.groupby('date',sort=False).close.transform(lambda z:z.ewm(span=5,adjust=False).mean())
    b5['ema10']=b5.groupby('date',sort=False).close.transform(lambda z:z.ewm(span=10,adjust=False).mean())
    rng=b5.high-b5.low; b5['body']=(b5.close-b5.open).abs()/rng.replace(0,np.nan)
    lastday=x.groupby('date',sort=False).close.last(); b5['prev_close']=b5.date.map(lastday.shift(1))
    b5['oc945']=b5.date.map(b5[b5.bucket<=5].groupby('date',sort=False).close.last())
    b5['ph1']=b5.groupby('date',sort=False).high.shift(1); b5['pl1']=b5.groupby('date',sort=False).low.shift(1)
    cand=[]
    for r in b5.itertuples(index=False):
        if r.bucket<6 or r.bucket>65 or pd.isna(r.prev_close):continue
        ei=int(r.last_idx)+1
        if ei>=N or D[ei]!=D[int(r.last_idx)]:continue
        if r.oc945>r.prev_close:d=1
        elif r.oc945<r.prev_close:d=-1
        else:continue
        if d==1 and not (r.ema5>r.ema10):continue
        if d==-1 and not (r.ema5<r.ema10):continue
        if d15_m[ei]!=0 and d15_m[ei]!=d:continue
        ph1=float(r.ph1) if pd.notna(r.ph1) else np.nan; pl1=float(r.pl1) if pd.notna(r.pl1) else np.nan
        # 14-column shape retained for the compiled simulator; unused fields are zero.
        cand.append((ei,int(r.last_idx),d,int(r.bucket),float(r.close),0.,0.,ph1,pl1,0.,0.,0.,0.,float(r.body)))
    return np.array(cand,float) if cand else np.empty((0,14),float)

def _r7_morning_layer(df,base,ctx):
    state=_r7_state(df,base,False); ca=_r7_morning_candidates(ctx)
    oe,ox,op,od=_r7_morning_sim_fast(ctx['O'],ctx['H'],ctx['L'],ctx['C'],ctx['D'],ctx['W'],state['occ'].astype(np.int8),state['bst'],ctx['day_end'],ctx['d15_m'],state['live_day'],state['live_week'],state['bext'],state['bp'],state['nextbst'],ca)
    z=_r7_simple_layer('R7_7_MORNING_PROOF_LONG',ctx['T'],oe,ox,op,od)
    return z

def _r7_combine(base,*layers):
    parts=[base]
    for z in layers:
        if z is not None and len(z):parts.append(z[['Module','Direction','Entry_Time','Exit_Time','Points']].copy())
    q=pd.concat(parts,ignore_index=True);q.Entry_Time=pd.to_datetime(q.Entry_Time);q.Exit_Time=pd.to_datetime(q.Exit_Time)
    return q.sort_values(['Exit_Time','Entry_Time','Module']).reset_index(drop=True)


def run_engine(args):
    init(args.nifty)
    p=r4_core_params(args.start,args.end)
    _,legs,masters=sim(p,frames=True)
    core=_core_frame(legs,masters)
    sdf=DF[(DF.timestamp>=pd.Timestamp(args.start))&(DF.timestamp<pd.Timestamp(args.end))].copy().reset_index(drop=True)
    swing=run_r3_swing(sdf,core)
    feat=_build_repair_features(sdf)
    base=pd.concat([core[['Entry_Time','Exit_Time','Points']].assign(Module='CORE'),swing[['Entry_Time','Exit_Time','Points']].assign(Module='SWING')],ignore_index=True)
    repair_a=_run_week_repair(sdf,base,feat,_r3_repair_a_config(),'REPAIR_A')
    with_a=pd.concat([base,repair_a[['Entry_Time','Exit_Time','Points','Module']]],ignore_index=True)
    repair_b=_run_week_repair(sdf,with_a,feat,_r3_repair_b_config(),'REPAIR_B')
    r4=_combine_generated(core,swing,repair_a,repair_b)
    F=_r5_session_features(sdf,r4)
    sigA=_r5_build_signal(F,4,40.0,0.60,0.0,11)
    A=_r5_sim_breakout(F,*sigA,F['occ'],F['start'],F['live_day'],F['live_week'],20.,45.,70.,20.,4,999.,10.,10.,20.,'R5_A')
    occA,stA,dayA,weekA,drA,wrA=_r5_add_layer(F,A,F['occ'],F['start'],F['live_day'],F['live_week'],F['day_real'],F['week_real'])
    sigB=_r5_build_signal(F,3,0.0,0.30,0.0,15);sigB=_r6_filter_b_midday_entries(F,sigB)
    B5=_r5_sim_breakout(F,*sigB,occA,stA,dayA,weekA,999.,999.,999.,20.,2,20.,1.,1.,20.,'R5_B')
    occB,stB,dayB,weekB,drB,wrB=_r5_add_layer(F,B5,occA,stA,dayA,weekA,drA,wrA)
    tr,cross,br5=_r5_c2145_arrays(F);C15=_r5_sim_c2145(F,occB,stB,tr,cross,br5)
    r6=_r7_detail_from_r6(core,swing,repair_a,repair_b,A,B5,C15)
    ctx=_r7_common_context(sdf);ctx['df']=sdf
    # R7 sequence: each layer sees only prior generated current-run trades.
    r71=_r7_layer1(sdf,r6,ctx); b71=_r7_combine(r6,r71)
    r72=_r7_layer2(sdf,b71,ctx); b72=_r7_combine(b71,r72)
    cfg3={'expansion_min':2.10,'body_min':.55,'week_min':-5.,'week_max':0.,'loss_streak_min':2,'loss_streak_max':3,'target':35.,'max_per_day':1,'exclude_1':(765,840),'exclude_2':(870,930),'be_mfe':15.,'lock_mfe':30.,'lock_points':5.}
    r73=_r7_sim_micro(ctx,b72,cfg3,'R7_3_STRONG_MICRO');b73=_r7_combine(b72,r73)
    cfg4={'expansion_min':2.05,'body_min':.60,'breakout_extension_min':.05,'week_min':-10.,'week_max':0.,'loss_streak_min':0,'loss_streak_max':2,'target':40.,'max_per_day':1,'exclude_1':(765,840),'exclude_2':(870,930),'be_mfe':15.,'lock_mfe':30.,'lock_points':5.}
    r74=_r7_sim_micro(ctx,b73,cfg4,'R7_4_STRONG_CHILD');b74=_r7_combine(b73,r74)
    cfg5={'expansion_min':1.90,'expansion_max':2.05,'body_min':.60,'breakout_extension_min':.10,'prior3_compression_max':1.50,'week_min':-10.,'week_max':0.,'loss_streak_min':0,'loss_streak_max':2,'target':40.,'max_per_day':1,'exclude_1':(765,840),'exclude_2':(870,930),'be_mfe':15.,'lock_mfe':30.,'lock_points':5.}
    r75=_r7_sim_micro(ctx,b74,cfg5,'R7_5_INTERMEDIATE');b75=_r7_combine(b74,r75)
    cfg6={'expansion_min':1.90,'body_min':.55,'ema_gap_norm_min':.10,'breakout_extension_min':.10,'min_since_1p9':20,'week_min':-10.,'week_max':0.,'loss_streak_min':1,'loss_streak_max':3,'target':25.,'max_per_day':2,'exclude_1':(765,840),'exclude_2':(870,930),'be_mfe':15.,'lock_mfe':30.,'lock_points':5.}
    r76=_r7_sim_micro(ctx,b75,cfg6,'R7_6_SECOND_CHILD');b76=_r7_combine(b75,r76)
    r77=_r7_morning_layer(sdf,b76,ctx);b77=_r7_combine(b76,r77)
    cfg8={'include_start_min':840,'include_end_min':870,'expansion_min':1.60,'body_min':.55,'ema_gap_norm_min':.15,'breakout_extension_min':.03,'week_min':-10.,'week_max':0.,'loss_streak_min':0,'loss_streak_max':2,'target':20.,'max_per_day':1,'be_mfe':15.,'lock_mfe':30.,'lock_points':5.}
    r78=_r7_sim_micro(ctx,b77,cfg8,'R7_8_LATE_CHILD');final=_r7_combine(b77,r78)
    fm=_r7_metrics_detail(final);counts=_r7_period_counts(sdf,final)
    yearly=[];yy=pd.to_datetime(final.Exit_Time).dt.year
    for y in sorted(yy.unique()):
        z=final[yy==y];yearly.append({'year':int(y),'points':float(z.Points.sum()),'trades':int(len(z))})
    layers={}
    for name,z in [('R7_1_IMPULSE',r71),('R7_2_QUIET_PARENT',r72),('R7_3_STRONG_MICRO',r73),('R7_4_STRONG_CHILD',r74),('R7_5_INTERMEDIATE',r75),('R7_6_SECOND_CHILD',r76),('R7_7_MORNING_PROOF_LONG',r77),('R7_8_LATE_CHILD',r78)]:
        layers[name]=_r7_metrics_detail(z) if z is not None and len(z) else {'net':0.,'pf':None,'dd':0.,'trades':0,'win_rate':0.}
    summary={'engine':'NIFTY_R7_RULES_ONLY_ITM3_SHORT_CHILDREN','combined':fm,'daily':counts['daily'],'weekly':counts['weekly'],'yearly':yearly,'r7_layers':layers,'rules':R7_RULES,
             'integrity':{'raw_minute_only':True,'historical_trade_rows_embedded':False,'historical_timestamps_embedded':False,'precomputed_pnl_embedded':False,'date_specific_fixes':False,'completed_bar_next_minute_fills':True,'minimum_paid_initial_stop_points':10.0}}
    print(json.dumps(summary,indent=2,default=str))
    if args.out:
        from pathlib import Path
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        core.to_csv(out/'core_generated.csv',index=False);swing.to_csv(out/'swing_generated.csv',index=False);repair_a.to_csv(out/'repair_a_generated.csv',index=False);repair_b.to_csv(out/'repair_b_generated.csv',index=False);A.to_csv(out/'r5_layer_a_generated.csv',index=False);B5.to_csv(out/'r5_layer_b_generated.csv',index=False);C15.to_csv(out/'r5_layer_c2145_generated.csv',index=False)
        for name,z in [('r7_1_impulse_generated.csv',r71),('r7_2_quiet_parent_generated.csv',r72),('r7_3_strong_micro_generated.csv',r73),('r7_4_strong_child_generated.csv',r74),('r7_5_intermediate_generated.csv',r75),('r7_6_second_child_generated.csv',r76),('r7_7_morning_proof_long_generated.csv',r77),('r7_8_late_child_generated.csv',r78)]: z.to_csv(out/name,index=False)
        final.to_csv(out/'all_generated.csv',index=False);(out/'summary.json').write_text(json.dumps(summary,indent=2,default=str))


def main():
    import argparse
    ap=argparse.ArgumentParser(description='Standalone rules-only NIFTY R7 engine')
    ap.add_argument('--nifty',required=True,help='raw NIFTY 1-minute CSV or ZIP')
    ap.add_argument('--start',required=True);ap.add_argument('--end',required=True,help='exclusive end')
    ap.add_argument('--out',default=None,help='optional output directory for generated runtime reports')
    run_engine(ap.parse_args())

if __name__=='__main__': main()
