from pandas import DataFrame


class HilevLists:
    IDs = []
    TSTs = []
    WASOs = []
    SEs = []
    SFs = []
    SOLs = []

    def clear(self):
        self.IDs.clear()
        self.TSTs.clear()
        self.WASOs.clear()
        self.SEs.clear()
        self.SFs.clear()
        self.SOLs.clear()

    def to_data_frame(self):
        cols = ['ID', 'TST', 'WASO', 'SE', 'SF', 'SOL']
        data = {
            'ID': self.IDs,
            'TST': self.TSTs,
            'WASO': self.WASOs,
            'SE': self.SEs,
            'SF': self.SFs,
            'SOL': self.SOLs
        }
        return DataFrame(data, columns=cols)
