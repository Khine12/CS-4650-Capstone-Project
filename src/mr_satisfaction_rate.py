"""MapReduce job (Topics 4 + 9): satisfaction rate per Class x Type of Travel.

Run locally from the repo root:
    python src/mr_satisfaction_rate.py -r local data/train.csv

Average is the Topic 4 "average trap": partial results must be [sum, count],
not averages, so the combiner can run on any machine and still be correct.
"""
from mrjob.job import MRJob

# Column positions in train.csv (header checked 2026-11-26)
TYPE_OF_TRAVEL = 20
CLASS = 21
SATISFACTION = 22


class MRSatisfactionRate(MRJob):

    def mapper(self, _, line):
        fields = line.strip().split(",")
        if len(fields) != 23 or fields[0] == "id":   # skip header / bad lines
            return
        key = f"{fields[CLASS]} | {fields[TYPE_OF_TRAVEL]}"
        satisfied = 1 if fields[SATISFACTION] == "True" else 0
        yield key, [satisfied, 1]                    # [sum, count]

    def combiner(self, key, values):
        total, count = 0, 0
        for s, c in values:
            total += s
            count += c
        yield key, [total, count]                    # same shape as mapper

    def reducer(self, key, values):
        total, count = 0, 0
        for s, c in values:
            total += s
            count += c
        yield key, {"passengers": count, "satisfied_rate": round(total / count, 4)}


if __name__ == "__main__":
    MRSatisfactionRate.run()
