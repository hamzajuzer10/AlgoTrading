import pandas as pd
import numpy as np
import random

# set random seed
random.seed(1000)

# set global parameters
probability_win = 0.5
total_bank_roll = 20000
unit_size = 20
house_limit = 10000
iterations = 365

def single_round_outcome(probability_win: float, bank_roll: float, unit_size: float, house_limit:float):

    # set profit to 0
    profit = 0
    betsize = unit_size
    rounds = 0
    goal = 0
    results = pd.DataFrame()

    while (profit < unit_size) and (betsize <= house_limit) and (bank_roll >= betsize-profit):

        # update rounds
        rounds = rounds+1

        # generate a win or loss
        win = np.random.choice(np.arange(0, 2), p=[1-probability_win, probability_win])

        current_betsize = betsize

        if win:
            profit = profit + betsize

            if profit >= unit_size:

                goal = 1

            else:

                if profit + betsize >= unit_size:

                    betsize = unit_size - profit

                else:

                    betsize = betsize + unit_size

        else:

            profit = profit - betsize

        # store metrics
        metrics = {'goal': goal, 'win': win, 'profit': profit, 'betsize': current_betsize, 'rounds': rounds, 'unit_size': unit_size}
        results = results.append(metrics, ignore_index=True)

    # check if we hit the goal

    return results


def multi_round_outcome(iterations: int, probability_win: float, bank_roll: float, unit_size: float, house_limit:float):

    original_bank_roll = bank_roll
    total_win = 0
    average_win = 0
    average_bet_size = 0
    average_rounds = 0
    overall_profits = 0
    rounds_before_liquidation = iterations
    profit_profile = pd.DataFrame()


    # run 10k rounds

    for i in range(iterations):
        profile = {'profit': overall_profits}
        profit_profile = profit_profile.append(profile, ignore_index=True)
        result = single_round_outcome(probability_win=probability_win,
                                      bank_roll=bank_roll,
                                      unit_size=unit_size,
                                      house_limit=house_limit)

        if result.goal.max() == 0:

            # set new bank_roll and unit_size
            bank_roll = min(overall_profits, original_bank_roll)
            unit_size = (bank_roll/original_bank_roll)*unit_size
            overall_profits = overall_profits - bank_roll

            if bank_roll <= 0:
                rounds_before_liquidation = min(rounds_before_liquidation, i+1)
                break

        else:
            overall_profits = overall_profits + unit_size

        # compute mean metrics
        total_win = total_win + result.goal.max()
        average_win = average_win + (result.goal.max() - average_win) / (i + 1)
        average_bet_size = average_bet_size + (result.betsize.mean() - average_bet_size) / (i + 1)
        average_rounds = average_rounds + (result.rounds.mean() - average_rounds) / (i + 1)

    return total_win, average_win, average_bet_size, average_rounds, rounds_before_liquidation, overall_profits, profit_profile


if __name__ == '__main__':

    total_win, average_win, average_bet_size, average_rounds, \
    rounds_before_liquidation, overall_profits, profit_profile = multi_round_outcome(iterations=iterations,
                                                                                   probability_win=probability_win,
                                                                                   bank_roll=total_bank_roll,
                                                                                   unit_size=unit_size,
                                                                                   house_limit=house_limit)
    if (average_win<1):
        overall_profits = overall_profits-total_bank_roll

    print("Done")