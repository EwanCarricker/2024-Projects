# NFL Project: Model to Predict Success Against The Spread

## Using 6 metrics, this model aims to predict teams that will cover the spread in NFL games

#### Overview of the problem

The spread is a term used to ascertain which team is favoured to win each game. If a team is the favourite, then they will have to beat the opposition by a certain amount to cover. If a team is an underdog, then they just have to win or lose by smaller than the specified spread.

Here's an example from a recent game:
24/11/2024 Las Vegas Raiders (+5.5) [19]  vs Denver Broncos (-5.5) [29]

The brackets show the spread and the actual score. Denver had to cover by 5.5 and they did so by 10, so they covered the spread. Las Vegas had to win, or lose by less than 6 and so, given they lost by 10, they didn't cover. Only one team can cover each game.

#### NFL Game Data

I downloaded the NFL data from 'https://www.aussportsbetting.com/data/'. I took the data from the 2014-2024 seasons - this is ~2800 games. This data included column headers such as:
* Date of Game
* Home/Away Team
* Home/Away Score
* Home/Away Line Open (the spread ~week before the game)
* Overtime

#### Metrics for Model
As an avid watcher of the sport, these are the parameters that I felt would have an impact of whether teams would cover or not:
* Home/Away cover %'s
* Distance travelled by the away team
* If either of the teams went to overtime in the previous game
* The historic cover rate of each teams spread (+6,-3 etc)
* The gap between games paired with a Win/Loss in the previous game
* Teams that play in domes/hot climates going away to cold climate stadiums

#### Future Enhancements
* Incorporate external weather and venue data to improve predictions
* More team-by-team focused
* Include data on head coaches / quarterbacks

#### Replicating this code
git clone [(https://github.com/EwanCarricker/2024/tree/main/NFL_model]

* RUN_MODEL.py: Runs the program for the next week's fixtures/
* future_fixtures.py: Pooled together the metrics and calculated the best weighting.
* NFL2.py: Basis code for each metric.
* NFL.xlsx: Raw Data File (you will need to update this each week)

##### Runnning the Model
Open cmd prompt and type "python RUN_MODEL.py"
After a couple of minutres, this will return the Home Team, Away Team, Home Cover %, Away Cover %, Cover?
