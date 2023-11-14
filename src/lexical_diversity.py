import random
import collections
import numpy as np
from scipy.optimize import curve_fit
from lexical_diversity import lex_div as ld


class LexicalDiversity:
    """
    Converting text to charachters for lexical measures.
    """

    def __init__(self, text):
        self.text = text
        self.tokens = list(self.text)

    def _lexical_diversity(self):
        ttr = ld.ttr(self.tokens)
        root_ttr = ld.root_ttr(self.tokens)
        log_ttr = ld.log_ttr(self.tokens)
        maas_ttr = ld.maas_ttr(self.tokens)
        msstr = ld.msttr(self.tokens)
        mattr = ld.mattr(self.tokens)
        hdd = ld.hdd(self.tokens)
        mtld = ld.mtld(self.tokens)
        mtld_ma = ld.mtld_ma_wrap(self.tokens)
        mtld_ma_bid = ld.mtld_ma_bid(self.tokens)
        # VOC-D measure
        D_init = 0
        for _ in range(3):
            D_init = D_init + self.vocd(self.tokens)
        D = D_init / 3

        yulesK = self.get_yules(self.tokens)

        return {
            "TTR": ttr,
            "Root TTR": root_ttr,
            "Log TTR": log_ttr,
            "Maas TTR": maas_ttr,
            "Msstr": msstr,
            "Ma TTR": mattr,
            "HDD": hdd,
            "MTLD": mtld,
            "MTLD MA": mtld_ma,
            "MTLD MA Bi": mtld_ma_bid,
            "VocD": D,
            "YulesK": yulesK,
        }

    def safe_divide(self, numerator, denominator):
        index = 0 if (denominator == 0 or denominator) else numerator / denominator
        return index

    def vocd(self, token):
        try:
            average_ttr = []
            for sample in range(35, 51):
                ttr_ = []
                for _ in range(100):
                    sampleTokens = random.sample(token, sample)
                    tt = ld.ttr(sampleTokens)
                    ttr_.append(tt)
                avg = np.mean(ttr_)
                average_ttr.append(avg)
            N = [x for x in range(35, 51)]
            popt, _ = curve_fit(self.func, N, average_ttr)

            return popt[0]
        except:
            return 0

    def func(self, n, d):
        ttrr = (d / n) * (np.sqrt(1 + 2 * (n / d)) - 1)
        return ttrr

    def get_yules(self, token):
        token_counter = collections.Counter(tok for tok in token)
        m1 = sum(token_counter.values())
        m2 = sum([freq**2 for freq in token_counter.values()])
        i = (m1 * m1) / (m2 - m1)
        k = 1 / i * 10000
        return k
