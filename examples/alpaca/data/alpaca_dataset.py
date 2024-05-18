# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import h5py as h5
from functools import lru_cache

import numpy as np
import seaborn as sns
import pandas as pd
import torch
import multiprocessing
from itertools import repeat
from fairseq.file_io import PathManager

from fairseq.data import FairseqDataset


def getitem__worker(fname, fInds, self):
    means = {}
    # fInds = fInds[:5]
    with h5.File(fname, mode="r+") as df:
        try:
            del df["general/percies"]
        except Exception as e:
            print(e)
        try:
            del df["general/classWeights"]
        except Exception as e:
            print(e)
        try:
            del df["general/means"]
        except Exception as e:
            print(e)
        dfSymbols = [sbl for sbl in list(df.keys()) if "general" not in sbl]
        df.create_dataset(name="general/percies", data=(np.ones((len(dfSymbols), )) * 10))
        df.create_dataset(name="general/classWeights", data=np.ones((len(dfSymbols), 3)))
        for smbl in dfSymbols:
            if smbl not in means.keys():
                means[smbl] = []
            means[smbl].append(df[smbl][:, :2].flatten())

        for smbl in means.keys():
            m = np.concatenate(means[smbl])
            means[smbl] = m[m != -1].mean()
        means = np.array([means[smbl] for smbl in means.keys()])

    jumps = [self.itemgetter(i)[3] for i in fInds]
    print(len(jumps))
    jumps = torch.stack(jumps, dim=0)

    percies = np.percentile(jumps, 95, axis=0)

    with h5.File(fname, mode="r+") as df:
        dfSymbols = [sbl for sbl in list(df.keys()) if "general" not in sbl]
        df["general/percies"][:] = percies
        df.create_dataset(name="general/means", data=means)

    tgt = [self.itemgetter(i)[1] for i in fInds]
    tgt = sum(tgt)
    tgt = tgt.reshape((-1, self.klineSz))[:, :3]
    tgt /= len(self)

    with h5.File(fname, mode="r+") as df:
        df["general/classWeights"][:] = tgt


class AlpacaDataset(FairseqDataset):

    def __init__(self, path, batchSize, tokensPerSample, futureLenMins, nSymbols, klineSz):
        super().__init__()
        self.path = path
        self.nSymbols = nSymbols
        self.klineSz = klineSz
        self.genDataPath = os.path.join(self.path, "generalData.h5")
        self.tokensPerSample = tokensPerSample
        self.futureLenMins = futureLenMins
        self.batchSize = batchSize

        self._len = 0
        self.shapes = []
        self.files = [os.path.join(self.path, f, "cryptoBars.h5") for f in os.listdir(self.path) if "generalData" not in f]
        for fname in self.files:
            with h5.File(fname, mode="r") as df:
                dfSymbols = [sbl for sbl in list(df.keys()) if "general" not in sbl]
                self.shapes.append(df[dfSymbols[0]].shape)
                self._len += self.shapes[-1][0] - self.futureLenMins
        self.itemShape = (self.tokensPerSample, self.nSymbols * self.shapes[0][-1])
        self.tgtItemShape = self.itemShape
        # self.calcGeneralData()

    def printGeneralData(self):
        for fname in self.files:
            with h5.File(fname, mode="r+") as df:
                for key in ("percies", ):
                    # print(f"{key}:")
                    print(df["general"][key][:].max())
                    # pdf = pd.DataFrame()

    def calcGeneralData(self):
        currInd = 0
        inds = []
        for shape in self.shapes:
            inds.append(range(currInd, currInd + shape[0] - self.futureLenMins))
            currInd += shape[0] - self.futureLenMins
        #for f, ixs in zip(self.files, inds):
        #    for i in ixs:
        #        fc = self.getFileAndIdxFromMajorIdx(i)[1]
        #        if f != fc:
        #            raise Exception("indices are wrong")

        #for f, i, s in zip(self.files, inds, repeat(self)):
        #    getitem__worker(f, i, s)
        with multiprocessing.Pool(16) as p:
            list(p.starmap(getitem__worker, zip(self.files, inds, repeat(self))))

    def __del__(self):
        pass

    def __len__(self) -> int:
        return int(self._len)

    def size(self, index):
        return self.tokensPerSample

    def get_batch_shapes(self):
        return [(self.batchSize, self.tokensPerSample)]

    def num_tokens(self, index):
        return self.tokensPerSample

    def __frameGetter(self, firstFrameIdx, lastFrameIdx, data, ttlSz):
        if firstFrameIdx < 0:
            _frame = data[:lastFrameIdx + self.futureLenMins]
            _frame[:, -2] /= 10000  # downsize volume to prevent infs after quantizations
            _frame[:, -2] -= 50000
            _frame[_frame > 50000] = 50000
            frame = np.concatenate((-np.ones((-firstFrameIdx, data.shape[-1]), dtype=np.float32), _frame), axis=0)
        else:
            frame = data[firstFrameIdx:lastFrameIdx + self.futureLenMins]
            frame[:, -2] /= 10000
            frame[:, -2] -= 50000
            frame[frame > 50000] = 50000
        return frame[:-self.futureLenMins], frame[-self.futureLenMins:]

    def getFileAndIdxFromMajorIdx(self, idx):
        citer = 0
        for file, shape in zip(self.files, self.shapes):
            shp = shape[0] - self.futureLenMins
            citer += shp
            if citer > idx:
                return shape, file, idx - (citer - shp)

    def __getTargets(self, frame, fframe, ulMargin, llMargin):
        # frame[:, 0] = open
        # frame[:, 1] = close
        # frame[:, 2] = high
        # frame[:, 3] = low
        # frame[:, 4] = volume
        # frame[:, 5] = time stamp

        # return torch.tensor([[0, 1, 0]], dtype=torch.float16), torch.tensor([0.0], dtype=torch.float16)
        
        # marginIdx = next(i for i, x in enumerate(tgtSmblInd) if x == smblIdx)
        _fframe = fframe.swapaxes(0, 1)[..., :2].reshape((fframe.shape[1], -1))
        present = frame[self.tokensPerSample - 1].reshape((fframe.shape[1], -1))[:, 1]
        _max, amax = torch.max(_fframe, dim=-1)
        _min, amin = torch.min(_fframe, dim=-1)
        maskShort = ((present - _min) > ulMargin) & (((_max - present) < llMargin) | amax > amin)
        maskLong = ((_max - present) > ulMargin) & (((present - _min) < llMargin) | amin > amax)
        maskIdle = maskShort & maskLong
        maskIdle = maskIdle | ((maskShort == False) & (maskLong == False))
        maskShort = (maskIdle == False) & maskShort
        maskLong = (maskIdle == False) & maskLong
        tgt = torch.stack((maskLong, maskIdle, maskShort), dim=-1).to(torch.float32)
        # diff = _max - _min
        # tgt_ = tgt.argmax()
        # if tgt_ == 1 and diff > 5:
        #     pass
        # print(f"{tgt_} ; {diff}")

        m1 = (amax > amin) & (_min < present)
        m2 = (amin > amax) & (_max > present)
        m3 = (m1 == False) & (m2 == False)
        m = torch.stack((m1, m2, m3), dim=-1)
        jump = torch.stack((present - _min, _max - present, torch.zeros_like(present)), dim=-1)[m]
        return tgt, jump.to(torch.float16)

    def __getFutureTickLenFwd(self, data, presentIdx):
        futureLenSecs = self.futureLenMins * 60
        currentLenSecs = 0
        for idx in range(self._len):
            _idx = (presentIdx + idx) % self._len
            currentLenSecs += data[_idx][5]
            tickLen = _idx - presentIdx
            if tickLen < 0:
                tickLen = _idx - tickLen
            if currentLenSecs >= futureLenSecs:
                break
        return tickLen

    def __getFutureTickLenBwd(self, data, mostFutureIdx):
        futureLenSecs = self.futureLenMins * 60
        currentLenSecs = 0
        for idx in reversed(range(self._len)):
            _idx = (mostFutureIdx + idx) % self._len
            currentLenSecs += data[_idx][5]
            tickLen = mostFutureIdx - _idx
            if tickLen < 0:
                tickLen = mostFutureIdx - tickLen
            if currentLenSecs >= futureLenSecs:
                break
        return tickLen

    def itemgetter(self, idx):
        shape, file, presentIdx = self.getFileAndIdxFromMajorIdx(idx)
        dataFrames, fframes = [], []
        with h5.File(file) as df:
            percies = df["general/percies"][:]
            tgtSmblInd = (percies > 5).nonzero()[0]
            classWeights = torch.from_numpy((1 - df["general/classWeights"][:]) / 2)
            tgtSmblInd = torch.from_numpy(tgtSmblInd)
            tgtUpperMargins = percies
            ulMargin = torch.from_numpy(tgtUpperMargins)
            llMargin = torch.from_numpy(tgtUpperMargins / 5)

            dfSymbols = [sbl for sbl in list(df.keys()) if "general" not in sbl]
            pastIdx = presentIdx - self.tokensPerSample + 1
            for i, symbol in enumerate(dfSymbols):
                if symbol not in dfSymbols or presentIdx == 0:
                    dataFrames.append(- torch.ones(size=(self.tokensPerSample, shape[1]), dtype=torch.float16))
                    fframes.append(- torch.ones(size=(self.futureLenMins, shape[1]), dtype=torch.float16))
                else:
                    frame, fframe = self.__frameGetter(pastIdx, presentIdx, df[symbol], self._len)
                    frame = np.concatenate((frame, np.ones_like(frame, shape=(1,) + frame.shape[1:])), axis=0)  # append mask token
                    dataFrames.append(torch.from_numpy(frame))
                    fframes.append(torch.from_numpy(fframe))
                if dataFrames[-1].isinf().any().item() or dataFrames[-1].isnan().any().item():
                    raise Exception("PPAN: Found invalid val in dataloader")

        srcItem = torch.cat(dataFrames, dim=-1).to(torch.float16)
        future = torch.stack(fframes, dim=-2).to(torch.float16)
        # tgtItem, jumps = self.__getTargets(srcItem.to("cuda"), future.to("cuda"), ulMargin.to("cuda"), llMargin.to("cuda"))
        tgtItem, jumps = self.__getTargets(srcItem, future, ulMargin, llMargin)
        future = future.reshape((future.shape[0], -1)).cpu()
        tgtItem = tgtItem.cpu()
        padSz = shape[1] - tgtItem.shape[-1]
        tgtItem = torch.cat((tgtItem, torch.zeros(size=tgtItem.shape[:-1] + (padSz, ))), dim=-1)  # pad to src size
        return srcItem, tgtItem, future, jumps.cpu(), classWeights, tgtSmblInd

    @lru_cache(maxsize=8)
    def __getitem__(self, idx) -> torch.Tensor:
        return self.itemgetter(idx)

    def collater(self, samples):
        src = torch.stack([s[0] for s in samples], dim=0)
        tgt = torch.stack([s[1] for s in samples], dim=0)
        fut = torch.stack([s[2] for s in samples], dim=0)
        classWeights = torch.stack([s[3] for s in samples], dim=0)
        tgtSmblInd = torch.stack([s[4] for s in samples], dim=0)
        if src.isinf().any().item() or src.isnan().any().item() or tgt.isinf().any().item() or tgt.isnan().any().item():
            raise Exception("PPAN: Found invalid val in dataloader")
        return src, tgt, (fut, classWeights, tgtSmblInd)

    @staticmethod
    def exists(path):
        return PathManager.exists(path)

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


if __name__ == "__main__":
    pass
    AlpacaDataset('/home/drford/Documents/Projects/alpaca_trainer/TrainData/train', 128, 30, 10, 16, 8).printGeneralData()
    # AlpacaDataset('/home/drford/Documents/Projects/alpaca_trainer/TrainData/train', 128, 30, 10, 16, 8)
