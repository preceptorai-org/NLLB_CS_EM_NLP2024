import json
from collections import defaultdict
import random, math


# Elo rating retrieve
# 30 is the cutoff between  old and new method
INPUT_FILE_PATH = "QuestionaireAnswerCleaned.json"
with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
    elo_data = json.loads(f.read())


def to_filter_out(i, item):
    if item["sessionEndTimeSecond"].strip() == "":
        return True
    if int(item["sessionEndTimeSecond"]) - int(item["sessionStartTimeSecond"]) <= 300:
        print(
            "Warning session Time too short",
            i,
            int(item["sessionEndTimeSecond"]) - int(item["sessionStartTimeSecond"]),
        )
        return True
    job = item["job"]
    if job in {"Others", "No data"}:
        print("Warning Other job", i, job)
        return True
    return False


elo_data = [item for i, item in enumerate(elo_data) if not to_filter_out(i, item)]

win_pair = defaultdict(lambda: defaultdict(int))
all_models = set()
for record in elo_data:
    scoring_list = record["scoringData"]
    for scoring_data in scoring_list:
        ranking = list(
            zip(
                scoring_data["formState"]["ranking"],
                scoring_data["formState"]["translators"],
            )
        )
        ranking.sort()
        translator_ranked = [(item[1]) for item in ranking]

        for i in range(len(translator_ranked)):
            all_models.add(translator_ranked[i])
            for j in range(i + 1, len(translator_ranked)):
                win_pair[translator_ranked[i]][translator_ranked[j]] += 1


all_models = list(all_models)


def GLICKO_calc():
    r = [1500 for _ in all_models]
    RD = [350 for _ in all_models]
    q = math.log(10) / 400

    def g(rd):
        return (1 + 3 * ((q * rd / math.pi) ** 2)) ** (-0.5)

    def E(r, rj, rdj):
        return 1 / (1 + 10 ** ((-g(rdj) * (r - rj)) / 400))

    def d_sq_inv(i):
        s = 0
        for j in range(len(all_models)):
            win = win_pair[all_models[i]][all_models[j]]
            lose = win_pair[all_models[j]][all_models[i]]
            E_i_j = E(r[i], r[j], RD[j])
            g_j = g(RD[j])
            s += (win + lose) * g_j**2 * E_i_j * (1 - E_i_j)
        s *= q * q
        return s

    def new_rating(i):
        s = 0
        for j in range(len(all_models)):
            win = win_pair[all_models[i]][all_models[j]]
            lose = win_pair[all_models[j]][all_models[i]]
            E_i_j = E(r[i], r[j], RD[j])
            g_j = g(RD[j])
            s += g_j * (1 - E_i_j) * win + g_j * (0 - E_i_j) * lose
        denom = 1 / (RD[i] ** 2) + d_sq_inv(i)
        s *= q / (denom)
        new_r = r[i] + s
        new_rd = (denom) ** (-0.5)
        return (new_r, new_rd)

    new_ratings = [new_rating(i) for i in range(len(all_models))]
    return new_ratings


glicko = GLICKO_calc()
glicko_attached_model = list(zip(all_models, glicko))
glicko_attached_model.sort(reverse=True, key=lambda item: item[1][0])
for model_name, (rating, sd) in glicko_attached_model:

    print(
        "%.2f +- %.2f = [%.2f,%.2f] : %s"
        % (rating, 2 * sd, rating - 2 * sd, rating + 2 * sd, model_name)
    )

glicko_sorted_models = [item[0] for item in glicko_attached_model]
print(glicko_sorted_models)
for i in range(len(all_models)):
    sum_win = 0
    sum_lose = 0
    for j in range(len(all_models)):
        win = win_pair[glicko_sorted_models[i]][glicko_sorted_models[j]]
        lose = win_pair[glicko_sorted_models[j]][glicko_sorted_models[i]]
        print("%2d/%-2d" % (win, win + lose), end=" ")
        sum_win += win
        sum_lose += lose
    print("Win rate : ", sum_win / (sum_win + sum_lose + 1e-8))
