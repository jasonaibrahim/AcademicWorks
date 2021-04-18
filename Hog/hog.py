"""The Game of Hog"""
from math import *
from dice import four_sided_dice, six_sided_dice, make_test_dice
from ucb import main, trace, log_current_line, interact

goal = 100          # The goal of Hog is to score 100 points.
commentary = False  # Whether to display commentary for every roll.

# Taking turns

def roll_dice(num_rolls, dice=six_sided_dice, who='Boss Hogg'):
    """Calculate WHO's turn score after rolling DICE for NUM_ROLLS times.

    num_rolls:  The number of dice rolls that will be made; at least 1.
    dice:       A function of no args and returns an integer outcome.
    who:        Name of the current player, for commentary.
    """
    roll_total = 0
    got_a_one = False
    assert type(num_rolls) == int, 'num_rolls must be an integer.'
    assert num_rolls > 0, 'Must roll at least once.'

    for x in range (0,num_rolls):
        a = dice()
        if commentary:
            announce(a,who)
        if a == 1:
            got_a_one = True
        else:
            roll_total += a

    if got_a_one:
        return 1
    else:
        return roll_total


# Write your own prime functions here!
def is_prime(a):
    return a in primes()

def primes():
    return (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
            61, 67, 71, 73, 79, 83, 89, 97, 101)
            

def next_prime(a):
    for p in primes():
        if p > a:
            return p

# End prime functions here

def take_turn(num_rolls, opponent_score, dice=six_sided_dice, who='Boss Hogg'):
    """Simulate a turn in which WHO chooses to roll NUM_ROLLS, perhaps 0.

    num_rolls:       The number of dice rolls that will be made.
    opponent_score:  The total score of the opponent.
    dice:            A function of no args and returns an integer outcome.
    who:             Name of the current player, for commentary.
    """
    assert type(num_rolls) == int, 'num_rolls must be an integer.'
    assert num_rolls >= 0, 'Cannot roll a negative number of dice.'
    if commentary:
        print(who, 'is going to roll', num_rolls, 'dice')

    #Turn score
    turn_score = 0

    def free_bacon(opponent_score):
        return (opponent_score // 10 + 1) if (opponent_score >= 10) else 1

    #Roll the dice, keep track of the score, enforce free bacon rule
    if num_rolls == 0:
        turn_score = free_bacon(opponent_score)
    else:
        turn_score = roll_dice(num_rolls,dice,who)

    #Rules

    #Touchdown
    if turn_score % 6 == 0:
        turn_score += turn_score//6

    #Hogtimus Prime
    if is_prime(turn_score):
        turn_score = next_prime(turn_score)

    return turn_score


def take_turn_test():
    """Test the roll_dice and take_turn functions using test dice."""
    print('-- Testing roll_dice --')
    dice = make_test_dice(4, 6, 1)
    assert roll_dice(2, dice) == 10, 'First two rolls total 10'

    dice = make_test_dice(4, 6, 1)
    assert roll_dice(3, dice) == 1, 'Third roll is a 1'

    dice = make_test_dice(1, 2, 3)
    assert roll_dice(3, dice) == 1, 'First roll is a 1'

    print('-- Testing take_turn --')
    dice = make_test_dice(4, 6, 1)
    assert take_turn(2, 0, dice) == 10, 'First two rolls total 10'

    dice = make_test_dice(4, 6, 1)
    assert take_turn(3, 20, dice) == 1, 'Third roll is a 1'

    print('---- Testing Free Bacon rule ----')
    assert take_turn(0, 34) == 4, 'Opponent score 10s digit is 3'
    assert take_turn(0, 71) == 8, 'Opponent score 10s digit is 7'
    assert take_turn(0,  7) == 1, 'Opponont score 10s digit is 0'

    print('---- Testing Touchdown rule ----')
    dice = make_test_dice(6)
    assert take_turn(2, 0, dice) == 14, 'Original score was 12'
    assert take_turn(3, 0, dice) == 21, 'Original score was 18'

    print('---- Testing Hogtimus Prime rule ----')
    dice = make_test_dice(5, 6, 5, 2)
    assert take_turn(0, 42, dice) == 7, 'Opponent score 10s digit is 4'
    assert take_turn(2, 0, dice) == 13, 'Original score was 11'
    assert take_turn(0, 52, dice) == 11, 'Opponent score 10s digit is 5'
    assert take_turn(2, 0, dice) == 11, 'Original score was 7'

    print('Tests for roll_dice and take_turn passed.')

    '*** You may add more tests here if you wish ***'



# Commentator

def announce(outcome, who):
    """Print a description of WHO rolling OUTCOME."""
    print(who, 'rolled a', outcome)
    print(draw_number(outcome))

def draw_number(n, dot='*'):
    """Return a text representation of rolling the number N.
    If a number has multiple possible representations (such as 2 and 3), any
    valid representation is acceptable.

    >>> print(draw_number(3))
     -------
    |     * |
    |   *   |
    | *     |
     -------

    >>> print(draw_number(5))
     -------
    | *   * |
    |   *   |
    | *   * |
     -------

    >>> print(draw_number(6, '$'))
     -------
    | $   $ |
    | $   $ |
    | $   $ |
     -------
    """
    if n == 1:
        return draw_dice(1,0,0,0,dot)
    elif n == 2:
        return draw_dice(0,1,0,0,dot)
    elif n == 3:
        return draw_dice(1,1,0,0,dot)
    elif n == 4:
        return draw_dice(0,1,1,0,dot)
    elif n == 5:
        return draw_dice(1,1,1,0,dot)
    elif n == 6:
        return draw_dice(0,1,1,1,dot)
    else:
        return ''

def draw_dice(c, f, b, s, dot):
    """Return an ASCII art representation of a die roll.

    c, f, b, & s are boolean arguments. This function returns a multi-line
    string of the following form, where the letters in the diagram are either
    filled if the corresponding argument is true, or empty if it is false.

     -------
    | b   f |
    | s c s |
    | f   b |
     -------

    The sides with 2 and 3 dots have 2 possible depictions due to rotation.
    Either representation is acceptable.

    This function uses Python syntax not yet covered in the course.

    c, f, b, s -- booleans; whether to place dots in corresponding positions.
    dot        -- A length-one string to use for a dot.
    """
    assert len(dot) == 1, 'Dot must be a single symbol'
    border = ' -------'
    def draw(b):
        return dot if b else ' '
    c, f, b, s = map(draw, [c, f, b, s])
    top = ' '.join(['|', b, ' ', f, '|'])
    middle = ' '.join(['|', s, c, s, '|'])
    bottom = ' '.join(['|', f, ' ', b, '|'])
    return '\n'.join([border, top, middle, bottom, border])


# Game simulator

def num_allowed_dice(score, opponent_score):
    """Return the maximum number of dice allowed this turn. The maximum
    number of dice allowed is 10 unless the sum of SCORE and
    OPPONENT_SCORE has a 7 as its ones digit.

    >>> num_allowed_dice(1, 0)
    10
    >>> num_allowed_dice(5, 7)
    10
    >>> num_allowed_dice(7, 10)
    1
    >>> num_allowed_dice(13, 24)
    1
    """
    return 1 if ( (opponent_score+score == 7) or (opponent_score+score) % 10 == 7 ) else 10

def select_dice(score, opponent_score):
    """Select 6-sided dice unless the sum of scores is a multiple of 7.

    >>> select_dice(4, 24) == four_sided_dice
    True
    >>> select_dice(16, 64) == six_sided_dice
    True
    """
    return four_sided_dice if (opponent_score + score) % 7 == 0 else six_sided_dice

def other(who):
    """Return the other player, for players numbered 0 or 1.

    >>> other(0)
    1
    >>> other(1)
    0
    """
    return 1 - who

def name(who):
    """Return the name of player WHO, for player numbered 0 or 1."""
    if who == 0:
        return 'Player 0'
    elif who == 1:
        return 'Player 1'
    else:
        return 'An unknown player'

def play(strategy0, strategy1):
    """Simulate a game and return 0 if the first player wins and 1 otherwise.

    A strategy function takes two scores for the current and opposing players.
    It returns the number of dice that the current player will roll this turn.

    If a strategy returns more than the maximum allowed dice for a turn, then
    the maximum allowed is rolled instead.

    strategy0:  The strategy function for player 0, who plays first.
    strategy1:  The strategy function for player 1, who plays second.
    """
    who = 0 # Which player is about to take a turn, 0 (first) or 1 (second)
    score0 = 0
    score1 = 0

    while ((score0 < 100) and (score1 < 100)):

        if who == 0:

            x = strategy0(score0,score1)
            a = num_allowed_dice(score0,score1)

            if x > 10:
                score0 += take_turn(a, score1, select_dice(score0,score1), name(who))

            elif a == 1 and x >= 1:
                score0 += take_turn(1, score1, select_dice(score0,score1), name(who))

            elif a == 1 and x == 0:
                score0 += take_turn(0, score1, select_dice(score0,score1), name(who))

            else:
                score0 += take_turn(x, score1, select_dice(score0,score1), name(who))

            who = 1

        else:

            y = strategy1(score1,score0)
            b = num_allowed_dice(score1,score0)

            if y > 10:
                score1 += take_turn(b, score0, select_dice(score1,score0), name(who))

            elif b == 1 and y >= 1:
                score1 += take_turn(1, score0, select_dice(score1,score0), name(who))

            elif b == 1 and y == 0:
                score1 += take_turn(0, score0, select_dice(score1,score0), name(who))

            else:
                score1 += take_turn(y, score0, select_dice(score1,score0), name(who))

            who = 0

    if (score0 > score1):
        return 0
    else:
        return 1


# Basic Strategy

def always_roll(n):
    """Return a strategy that always rolls N dice.

    A strategy is a function that takes two game scores as arguments
    (the current player's score, and the opponent's score), and returns a
    number of dice to roll.

    If a strategy returns more than the maximum allowed dice for a turn, then
    the maximum allowed is rolled instead.

    >>> strategy = always_roll(5)
    >>> strategy(0, 0)
    5
    >>> strategy(99, 99)
    5
    """
    def strategy(score, opponent_score):
        return n
    return strategy


# Experiments (Phase 2)

def make_average(fn, num_samples=100):
    """Return a function that returns the average_value of FN when called.

    To implement this function, you will have to use *args syntax, a new Python
    feature introduced in this project.  See the project description.

    >>> dice = make_test_dice(3, 1, 5, 6)
    >>> avg_dice = make_average(dice, 100)
    >>> avg_dice()
    3.75
    >>> avg_score = make_average(roll_dice, 100)
    >>> avg_score(2, dice)
    6.0

    In this last example, two different turn scenarios are averaged.
    - In the first, the player rolls a 3 then a 1, receiving a score of 1.
    - In the other, the player rolls a 5 and 6, scoring 11.
    Thus, the average value is 6.0.
    """
    def avg_fn(*args):
        i = 0
        for x in range (0,num_samples):
            i += fn(*args)
        return i/num_samples
    return avg_fn

def compare_strategies(strategy, baseline=always_roll(5)):
    """Return the average win rate (out of 1) of STRATEGY against BASELINE."""
    as_first = 1 - make_average(play)(strategy, baseline)
    as_second = make_average(play)(baseline, strategy)
    return (as_first + as_second) / 2  # Average the two results

def eval_strategy_range(make_strategy, lower_bound, upper_bound):
    """Return the best integer argument value for MAKE_STRATEGY to use against
    the always-roll-5 baseline, between LOWER_BOUND and UPPER_BOUND (inclusive).

    make_strategy -- A one-argument function that returns a strategy.
    lower_bound -- lower bound of the evaluation range.
    upper_bound -- upper bound of the evaluation range.
    """
    best_value, best_win_rate = 0, 0
    value = lower_bound
    while value <= upper_bound:
        strategy = make_strategy(value)
        win_rate = compare_strategies(strategy)
        print('Win rate against the baseline using', value, 'value:', win_rate)
        if win_rate > best_win_rate:
            best_win_rate, best_value = win_rate, value
        value += 1
    return best_value


def run_experiments():
    """Run a series of strategy experiments and report results."""
    if False: # Change to False when done testing always_roll
        result = eval_strategy_range(always_roll, 1, 10)
        print('Best always_roll strategy:', result)

    if False: # Change to True when ready to test make_comeback_strategy
        result = eval_strategy_range(make_comeback_strategy, 5, 15)
        print('Best comeback strategy:', result)

    if True: # Change to True when ready to test make_mean_strategy
        result = eval_strategy_range(make_mean_strategy, 1, 10)
        print('Best mean strategy:', result)

    "*** You may add additional experiments here if you wish ***"


# Strategies

def make_comeback_strategy(margin, num_rolls=5):
    """Return a strategy that rolls one extra time when losing by MARGIN."""

    def s1(score,opponent_score):
        if ((opponent_score - score) >= margin):
            return num_rolls+1
        else:
            return num_rolls
    return s1

def make_mean_strategy(min_points, num_rolls=5):
    """Return a strategy that attempts to give the opponent problems."""

    def s2(score,opponent_score):
        x = take_turn(0,opponent_score)
        y = select_dice(x+score,opponent_score)
        z = num_allowed_dice(x+score,opponent_score)

        if (x >= min_points and (y == four_sided_dice or z == 1)):
            return 0
        else:
            return num_rolls
    return s2

def final_strategy(score, opponent_score):
    """Write a brief description of your final strategy.

    First we compute the expected values of rolling up to 10 four sided or six sided dice
    and enter them into an array whose index refers to the number of dice rolled. The maximum
    of the expected scores is the default number of dice to roll. 

    For certain circumstances, we
    apply strategies that enable us to use the special rules to our advantage. 

    We roll zero dice
    if the result of rolling zero allows us to reach the goal, or if we are in the lead and rolling
    zero dice gives a competitive advantage by using the rules against the opponent. We apply the same 
    procedure if the points scored from rolling zero dice is greater than a margin of 5 points. This
    value was taken from trials on make_mean_strategy and finding the margin with the highest win rate.
    
    If the opponent is in the lead, and the current player is about to roll a four sided dice, then
    a 3 is returned, since rolling a higher number of four sided dice has a higher probability of 
    rolling a 1. Otherwise if the opponent is in the lead by a certain margin, the current player rolls
    8 dice. This value is the highest acceptable risk, since the probability of not rolling a 1 with
    any number of rolls higher is very small. 

    If the current player is in the lead, then the current
    player rolls 3 dice, so as not to take more risks than necessary. For all other situations, we roll
    the default.
    """
    def E(n):
        """ Returns the expected score (without special rules applied) for rolling N six sided die
        """
        return pow((5/6),n)*4*n

    def E_4(n):
        """ Returns the expected score (without special rules applied) for rolling N four sided die
        """
        return pow((3/4),n)*3*n

    expected_scores = [] # array of expected values of scores. index refers to number of dice rolled
    d = select_dice(score,opponent_score) # which dice the current player will roll
    x = take_turn(0,opponent_score) # the points scored if the current player rolls 0 dice
    y = select_dice(x+score,opponent_score) # the dice the opponent must use if the current player rolls 0 dice
    z = num_allowed_dice(x+score,opponent_score) # the number of allowed dice the opponent will be allowed if the current player rolls 0 dice
    expected_scores.append(x) # simulate value of rolling zero dice and insert as first element of array
     # Fill in array of expected values
    for i in range(1,11):
        if d == six_sided_dice:
            expected_scores.append(floor(E(i)))
        else:
            expected_scores.append(floor(E_4(i)))

    m = max(expected_scores) # Find the maximum of the expected scores.

    if (x >= goal-score) or ((abs(score - opponent_score) < 12) and (y == four_sided_dice or z == 1)):
        return 0
    elif ((x >= 5) and (y == four_sided_dice or z == 1)):
        return 0
    elif ((opponent_score - score) >= 20) and (d == four_sided_dice):
        return 3
    elif (opponent_score - score) >= 20:
        return 8
    elif (score - opponent_score) >= 20:
        return 3
    else:
        return expected_scores.index(m) # Return the index of the maximum expected score.
    

def final_strategy_test():
    """Compares final strategy to the baseline strategy."""
    print('-- Testing final_strategy --')
    print('Win rate:', compare_strategies(final_strategy))


# Interaction.  You don't need to read this section of the program.

def interactive_strategy(score, opponent_score):
    """Prints total game scores and returns an interactive tactic.

    This function uses Python syntax/techniques not yet covered in this course.
    """
    print('Current score:', score, 'to', opponent_score)
    while True:
        response = input('How many dice will you roll? ')
        try:
            result = int(response)
        except ValueError:
            print('Please enter a positive number')
            continue
        if result < 0:
            print('Please enter a non-negative number')
        else:
            return result

def play_interactively():
    """Play one interactive game."""
    global commentary
    commentary = True
    print("Shall we play a game?")
    winner = play(interactive_strategy, always_roll(5))
    if winner == 0:
        print("You win!")
    else:
        print("The computer won.")

def play_basic():
    """Play one game in which two basic strategies compete."""
    global commentary
    commentary = True
    winner = play(always_roll(5), always_roll(6))
    if winner == 0:
        print("Player 0, who always wants to roll 5, won.")
    else:
        print("Player 1, who always wants to roll 6, won.")

@main
def run(*args):
    """Read in the command-line argument and calls corresponding functions.

    This function uses Python syntax/techniques not yet covered in this course.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Play Hog")
    parser.add_argument('--take_turn_test', '-t', action='store_true')
    parser.add_argument('--play_interactively', '-p', action='store_true')
    parser.add_argument('--play_basic', '-b', action='store_true')
    parser.add_argument('--run_experiments', '-r', action='store_true')
    parser.add_argument('--final_strategy_test', '-f', action='store_true')
    args = parser.parse_args()
    for name, execute in args.__dict__.items():
        if execute:
            globals()[name]()

