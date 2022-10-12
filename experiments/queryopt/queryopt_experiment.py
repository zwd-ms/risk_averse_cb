def shufflegen(gen, filepath, num_shuffles=1, bufsize=10000, seed=4545):
    import random

    state = random.Random(seed)
    for shuffle in range(num_shuffles):
        print(f"{shuffle=}")

        buf = [None] * bufsize

        for v in gen(filepath):
            index = state.randrange(bufsize)
            if buf[index] is not None:
                yield buf[index][0]

            buf[index] = (v,)

        for v in buf:
            if v is not None:
                yield v[0]


def loaddata(filepath):
    import csv
    import glob

    for filename in glob.glob(filepath):
        with open(filename, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter="|")
            header = next(reader)
            for _, row in enumerate(reader):
                rowdict = {k: v for k, v in zip(header, row)}

                context = {
                    k: int(v) for k, v in enumerate(rowdict["features"]) if v == "1"
                }
                actions = [{"ConfigId": id} for id in eval(rowdict["configs"])]
                rewards = eval(rowdict["rewards"])

                yield {
                    "context": context,
                    "actions": actions,
                    "rewards": rewards,
                }


def hash2vw(h):
    return "\n".join(
        ["shared |s " + " ".join([f"{k}:{v}" for k, v in h["context"].items()])]
        + [
            f"{dacost} |a " + " ".join([f"{k}={v}" for k, v in a.items()])
            for n, a in enumerate(h["actions"])
            for dacost in (
                (
                    f'0:{h["cost"]*.1}:{h["ppa"]}'
                    if ("playedaction" in h.keys()) and n == h["playedaction"]
                    else ""
                ),
            )
        ]
    )


def learn(q, gamma_scale=10, gamma_exponent=0.5, num_epochs=10, num_shuffles=10):
    from vowpalwabbit import pyvw
    import random

    filename = "queryopt_data.csv"
    vw = pyvw.Workspace(
        " ".join(
            [
                "-b 24 -q sa --cubic ssa --ignore_linear s",
                "--cb_explore_adf --cb_type mtr --squarecb",
                f"--gamma_scale {gamma_scale} --gamma_exponent {gamma_exponent}",
                f"--loss_function expectile --expectile_q {q}",
                "--quiet",
            ]
        )
    )

    for epoch in range(num_epochs):
        print(f"{epoch=}")
        sum_rewards = 0.0
        num_regress = 0
        sum_regress = 0

        for n, v in enumerate(shufflegen(loaddata, filename, num_shuffles)):
            exstring = hash2vw(v)
            pred = vw.predict(exstring)

            playedaction = random.choices(range(len(pred)), pred)[0]
            reward = v["rewards"][playedaction]
            sum_rewards += reward
            if reward < 0:
                num_regress += 1
                sum_regress += reward

            v["playedaction"] = playedaction
            v["ppa"] = pred[playedaction]
            v["cost"] = -reward

            exstring = hash2vw(v)
            vw.learn(exstring)

        print(f"num_examples={n+1}")
        print(f"{sum_rewards=}")
        print(f"{num_regress=}")
        print(f"{sum_regress=}\n")

    del vw
    assert sum_rewards > 0
    return sum_rewards, num_regress, sum_regress


def display(data):
    def mean_confidence_interval(data, confidence=0.95):
        import numpy as np
        import scipy.stats

        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return m, m - h, m + h

    m, lb, ub = mean_confidence_interval(data)
    print("avg,lb,ub,min,max")
    print(f"{m},{lb},{ub},{min(data)},{max(data)}\n")
    return m, lb, ub


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 5
    num_repeats = int(sys.argv[1])
    q = float(sys.argv[2])
    gamma_scale = float(sys.argv[3])
    gamma_exponent = float(sys.argv[4])

    rewards = []
    regress_cnt = []
    regress_mag = []
    for i in range(num_repeats):
        print(f"{q=} {i=}")
        sum_rewards, num_regress, sum_regress = learn(q, gamma_scale, gamma_exponent)
        rewards.append(sum_rewards)
        regress_cnt.append(num_regress)
        regress_mag.append(sum_regress / num_regress)

    result = [q]
    result += display(rewards)
    result += display(regress_cnt)
    result += display(regress_mag)
    print(",".join([str(f) for f in result]))
