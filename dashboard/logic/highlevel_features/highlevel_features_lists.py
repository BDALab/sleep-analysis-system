from pandas import DataFrame


class HilevLists:

    def __init__(self):
        self.IDs = []
        self.TIBs = []
        self.TSTs = []
        self.WASOs = []
        self.WASFs = []
        self.WBs = []
        self.SEs = []
        self.SFs = []
        self.SOLs = []
        self.WKS5 = []
        self.DTSTs = []

    def clear(self):
        self.IDs.clear()
        self.TIBs.clear()
        self.TSTs.clear()
        self.WASOs.clear()
        self.WASFs.clear()
        self.WBs.clear()
        self.SEs.clear()
        self.SFs.clear()
        self.SOLs.clear()
        self.WKS5.clear()
        self.DTSTs.clear()

    def to_data_frame(self):
        cols = ['ID', 'TIB', 'TST', 'WASO', 'WASF', 'WB', 'SE', 'SF', 'SOL', 'WKS5', 'DTST']
        data = {
            'ID': self.IDs,
            'TIB': self.TIBs,
            'TST': self.TSTs,
            'WASO': self.WASOs,
            'WASF': self.WASFs,
            'WB': self.WBs,
            'SE': self.SEs,
            'SF': self.SFs,
            'SOL': self.SOLs,
            'WKS5': self.WKS5,
            'DTST': self.DTSTs
        }
        return DataFrame(data, columns=cols)
