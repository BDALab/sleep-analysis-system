import logging
from datetime import timedelta
from os import path
from statistics import mean, median

import numpy
import pytz
from pandas import DataFrame

from dashboard.logic import cache
from dashboard.logic.machine_learning.settings import prediction_name, hilev_prediction
from dashboard.logic.sleep_diary.structure import create_structure
from dashboard.models import CsvData, SleepDiaryDay, SleepNight

logger = logging.getLogger(__name__)


def hilev():
    structure = create_structure()
    res = True
    IDs = []
    TSTs = []
    WASOs = []
    SEs = []
    SFs = []
    SOLs = []
    aIDs = []
    sTSTs = []
    sWASOs = []
    sSEs = []
    sSFs = []
    sSOLs = []
    aTSTs = []
    aWASOs = []
    aSEs = []
    aSFs = []
    aSOLs = []
    mIDs = []
    mTSTs = []
    mWASOs = []
    mSEs = []
    mSFs = []
    mSOLs = []
    ls = structure[0][0]
    i = 0
    for subject, data, day in structure:
        if not isinstance(data, CsvData) and path.exists(data.cached_prediction_path):
            res = False
            continue
        if not isinstance(day, SleepDiaryDay):
            res = False
            continue

        df = cache.load_obj(data.cached_prediction_path)
        if not isinstance(df, DataFrame):
            res = False
            continue
        nights = SleepNight.objects.filter(diary_day=day).filter(data=data).filter(subject=subject)
        if not nights.exists():
            night = SleepNight()
            night.diary_day = day
            night.data = data
            night.subject = subject
        else:
            night = nights.first()
        s = day.t1 - timedelta(minutes=30)
        e = day.t4 + timedelta(minutes=30)
        interval = df.loc[s:e, [prediction_name]]
        rolling_10 = interval.rolling('300s').sum()
        rolling_10['strict'] = numpy.where(rolling_10[prediction_name] <= 5, 'W', 'S')
        sleep = rolling_10.index[rolling_10['strict'] == 'S'].tolist()
        if not sleep:
            logger.warning(f'No sleep found for {night.subject.code} {night.diary_day.date} {night.data.filename}')
            res = False
            continue
        night.sleep_onset = pytz.timezone("Europe/Prague").localize(sleep[0])
        night.sleep_end = pytz.timezone("Europe/Prague").localize(sleep[-1])
        rolling_10[hilev_prediction] = numpy.where(rolling_10[prediction_name] <= 2, 'W', 'S')
        pred = rolling_10.loc[sleep[0]:sleep[-1], [hilev_prediction]]
        if not isinstance(pred, DataFrame):
            res = False
            continue
        pred.to_excel(night.name)
        wake = pred.index[pred[hilev_prediction] == 'W'].tolist()

        night.tst = (night.sleep_end - night.sleep_onset).seconds
        night.waso = len(wake) * 30
        night.se = ((night.tst - night.waso) / night.tst) * 100
        pred["number_prediction"] = numpy.where(pred[hilev_prediction] == 'S', 1, 0)
        wakes_counts = (pred["number_prediction"].diff() == -1).sum()
        night.sf = wakes_counts / (night.convert(night.tst).seconds / 3600)
        onset_latency = sleep[0] - day.t1 if sleep[0] > day.t1 else timedelta(seconds=0)
        night.sol = onset_latency.seconds
        logger.info(night)
        night.save()

        IDs.append(f'{subject.code}_{day.date}')
        TSTs.append(night.tst)
        WASOs.append(night.waso)
        SEs.append(night.se)
        SFs.append(night.sf)
        SOLs.append(night.sol)

        if ls != subject:
            aIDs.append(i)
            aTSTs.append(mean(sTSTs))
            aWASOs.append(mean(sWASOs))
            aSEs.append(mean(sSEs))
            aSFs.append(mean(sSFs))
            aSOLs.append(mean(sSOLs))
            mIDs.append(i)
            mTSTs.append(median(sTSTs))
            mWASOs.append(median(sWASOs))
            mSEs.append(median(sSEs))
            mSFs.append(median(sSFs))
            mSOLs.append(median(sSOLs))
            i += 1
            ls = subject
            sTSTs.clear()
            sWASOs.clear()
            sSEs.clear()
            sSFs.clear()
            sSOLs.clear()

        sTSTs.append(night.tst)
        sWASOs.append(night.waso)
        sSEs.append(night.se)
        sSFs.append(night.sf)
        sSOLs.append(night.sol)

    aIDs.append(i)
    aTSTs.append(mean(sTSTs))
    aWASOs.append(mean(sWASOs))
    aSEs.append(mean(sSEs))
    aSFs.append(mean(sSFs))
    aSOLs.append(mean(sSOLs))
    mIDs.append(i)
    mTSTs.append(median(sTSTs))
    mWASOs.append(median(sWASOs))
    mSEs.append(median(sSEs))
    mSFs.append(median(sSFs))
    mSOLs.append(median(sSOLs))
    i += 1
    ls = subject
    sTSTs.clear()
    sWASOs.clear()
    sSEs.clear()
    sSFs.clear()
    sSOLs.clear()

    cols = ['ID', 'TST', 'WASO', 'SE', 'SF', 'SOL']
    data = {
        'ID': IDs,
        'TST': TSTs,
        'WASO': WASOs,
        'SE': SEs,
        'SF': SFs,
        'SOL': SOLs
    }
    exp_df = DataFrame(data, columns=cols)
    exp_df.to_excel("hilevs.xlsx")

    adata = {
        'ID': aIDs,
        'TST': aTSTs,
        'WASO': aWASOs,
        'SE': aSEs,
        'SF': aSFs,
        'SOL': aSOLs
    }
    aexp_df = DataFrame(adata, columns=cols)
    aexp_df.to_excel("average_hilevs.xlsx")

    mdata = {
        'ID': mIDs,
        'TST': mTSTs,
        'WASO': mWASOs,
        'SE': mSEs,
        'SF': mSFs,
        'SOL': mSOLs
    }
    mexp_df = DataFrame(mdata, columns=cols)
    mexp_df.to_excel("median_hilevs.xlsx")

    return res
