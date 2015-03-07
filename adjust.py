__author__ = 'scottkellert'

import numpy as np
import pandas as pd
import operator
import scipy.stats as st

def teamInit(stats, team_list):
    # Initialize the rankings for each statistic
    # All teams begin with an equal rank of 100
    # The stats table comes from the detailed game statistics from kaggle
    # The team list is a list of all the team id's
    teams = {}
    for team in team_list.loc[:,'team_id']:
        if team in np.array(stats.loc[:,'wteam']):
            teams[team] = {'PTDIFF': 100, 'PF': 100, 'PA': 100, 'FGMF':100, 'FGMA':100,
                                'FGAF': 100, 'FGAA': 100, 'FGM3F': 100, 'FGM3A':100, 'FGA3F':100,
                                'FGA3A': 100, 'FTMF': 100, 'FTMA': 100, 'FTAF':100, 'FTAA':100,
                                'ORF': 100, 'ORA': 100, 'DRF': 100, 'DRA':100, 'ASTF':100,
                                'ASTA': 100, 'TOF': 100, 'TOA': 100, 'BLKF':100, 'BLKA':100,
                                'STLF': 100, 'STLA': 100, 'FF': 100, 'FA':100}
    return teams

def teamStatsInit(stats):
    # This function pulls each statistic for each team for each game
    # The output is a nested dictionary that allows for the look up of list of a team's
    # statistic over the course of a season
    # There are three situation handled in the conditional statements
    # Home, Away, and Neutral Court
    # Adjustments are made to reflect homecourt advantage
    # Finally if there are OT games we scale them to being a standard 40 min game
    teamStats = {}
    n = stats.shape[0]
    for g in range(n):
        if g % 500 == 0:
            print str(int(100*g/float(n)))+'%'
        wteam = stats.loc[g,'wteam']
        lteam = stats.loc[g,'lteam']
        if stats.loc[g,'wloc'] == 'H':
            wscore = stats.loc[g,'wscore'] - 4
            lscore = stats.loc[g,'lscore']
            wfgm = stats.loc[g,'wfgm'] - 1.5
            lfgm = stats.loc[g,'lfgm']
            wfga = stats.loc[g,'wfga']
            lfga = stats.loc[g,'lfga']
            wfgm3 = stats.loc[g,'wfgm3'] - .3
            lfgm3 = stats.loc[g,'lfgm3']
            wfga3 = stats.loc[g,'wfga3']
            lfga3 = stats.loc[g,'lfga3']
            wftm = stats.loc[g,'wftm'] - 2.4
            lftm = stats.loc[g,'lftm']
            wfta = stats.loc[g,'wfta'] - 3
            lfta = stats.loc[g,'lfta']
            wor = stats.loc[g,'wor'] - .4
            lor = stats.loc[g,'lor']
            wdr = stats.loc[g,'wdr'] - 1
            ldr = stats.loc[g,'ldr']
            wast = stats.loc[g,'wast'] - 2
            last = stats.loc[g,'last']
            wto = stats.loc[g,'wto'] + 1
            lto = stats.loc[g,'lto']
            wblk = stats.loc[g,'wblk'] - .5
            lblk = stats.loc[g,'lblk']
            wstl = stats.loc[g,'wstl'] - .5
            lstl = stats.loc[g,'lstl']
            wpf = stats.loc[g,'wpf'] + 1.5
            lpf = stats.loc[g,'lpf']
        elif stats.loc[g,'wloc'] == 'A':
            wscore = stats.loc[g,'wscore']
            lscore = stats.loc[g,'lscore'] - 4
            wfgm = stats.loc[g,'wfgm']
            lfgm = stats.loc[g,'lfgm'] - 1.5
            wfga = stats.loc[g,'wfga']
            lfga = stats.loc[g,'lfga']
            wfgm3 = stats.loc[g,'wfgm3']
            lfgm3 = stats.loc[g,'lfgm3'] - .3
            wfga3 = stats.loc[g,'wfga3']
            lfga3 = stats.loc[g,'lfga3']
            wftm = stats.loc[g,'wftm']
            lftm = stats.loc[g,'lftm'] - 2.4
            wfta = stats.loc[g,'wfta']
            lfta = stats.loc[g,'lfta'] - 3
            wor = stats.loc[g,'wor']
            lor = stats.loc[g,'lor'] - .4
            wdr = stats.loc[g,'wdr']
            ldr = stats.loc[g,'ldr'] - 1
            wast = stats.loc[g,'wast']
            last = stats.loc[g,'last'] - 2
            wto = stats.loc[g,'wto']
            lto = stats.loc[g,'lto'] + 1
            wblk = stats.loc[g,'wblk']
            lblk = stats.loc[g,'lblk'] - .5
            wstl = stats.loc[g,'wstl']
            lstl = stats.loc[g,'lstl'] - .5
            wpf = stats.loc[g,'wpf']
            lpf = stats.loc[g,'lpf'] - 1.5
        else:
            wscore = stats.loc[g,'wscore']
            lscore = stats.loc[g,'lscore']
            wfgm = stats.loc[g,'wfgm']
            lfgm = stats.loc[g,'lfgm']
            wfga = stats.loc[g,'wfga']
            lfga = stats.loc[g,'lfga']
            wfgm3 = stats.loc[g,'wfgm3']
            lfgm3 = stats.loc[g,'lfgm3']
            wfga3 = stats.loc[g,'wfga3']
            lfga3 = stats.loc[g,'lfga3']
            wftm = stats.loc[g,'wftm']
            lftm = stats.loc[g,'lftm']
            wfta = stats.loc[g,'wfta']
            lfta = stats.loc[g,'lfta']
            wor = stats.loc[g,'wor']
            lor = stats.loc[g,'lor']
            wdr = stats.loc[g,'wdr']
            ldr = stats.loc[g,'ldr']
            wast = stats.loc[g,'wast']
            last = stats.loc[g,'last']
            wto = stats.loc[g,'wto']
            lto = stats.loc[g,'lto']
            wblk = stats.loc[g,'wblk']
            lblk = stats.loc[g,'lblk']
            wstl = stats.loc[g,'wstl']
            lstl = stats.loc[g,'lstl']
            wpf = stats.loc[g,'wpf']
            lpf = stats.loc[g,'lpf']
        if stats.loc[g,'numot'] > 0:
            ot_adj = 40/float(40 + 5*stats.loc[g,'numot'])
            wscore *= ot_adj
            lscore *= ot_adj
            wfgm *= ot_adj
            lfgm *= ot_adj
            wfga *= ot_adj
            lfga *= ot_adj
            wfgm3 *= ot_adj
            lfgm3 *= ot_adj
            wfga3 *= ot_adj
            lfga3 *= ot_adj
            wftm *= ot_adj
            lftm *= ot_adj
            wfta *= ot_adj
            lfta *= ot_adj
            wor *= ot_adj
            lor *= ot_adj
            wdr *= ot_adj
            ldr *= ot_adj
            wast *= ot_adj
            last *= ot_adj
            wto *= ot_adj
            lto *= ot_adj
            wblk *= ot_adj
            lblk *= ot_adj
            wstl *= ot_adj
            lstl *= ot_adj
            wpf *= ot_adj
            lpf *= ot_adj
        try:
            teamStats[wteam]['Opp'].append(lteam)
            teamStats[wteam]['PF'].append(wscore)
            teamStats[wteam]['PA'].append(lscore)
            teamStats[wteam]['FGMF'].append(wfgm)
            teamStats[wteam]['FGMA'].append(lfgm)
            teamStats[wteam]['FGAF'].append(wfga)
            teamStats[wteam]['FGAA'].append(lfga)
            teamStats[wteam]['FGM3F'].append(wfgm3)
            teamStats[wteam]['FGM3A'].append(lfgm3)
            teamStats[wteam]['FGA3F'].append(wfga3)
            teamStats[wteam]['FGA3A'].append(lfga3)
            teamStats[wteam]['FTMF'].append(wftm)
            teamStats[wteam]['FTMA'].append(lftm)
            teamStats[wteam]['FTAF'].append(wfta)
            teamStats[wteam]['FTAA'].append(lfta)
            teamStats[wteam]['ORF'].append(wor)
            teamStats[wteam]['ORA'].append(lor)
            teamStats[wteam]['DRF'].append(wdr)
            teamStats[wteam]['DRA'].append(ldr)
            teamStats[wteam]['ASTF'].append(wast)
            teamStats[wteam]['ASTA'].append(last)
            teamStats[wteam]['TOF'].append(wto)
            teamStats[wteam]['TOA'].append(lto)
            teamStats[wteam]['BLKF'].append(wblk)
            teamStats[wteam]['BLKA'].append(lblk)
            teamStats[wteam]['STLF'].append(wstl)
            teamStats[wteam]['STLA'].append(lstl)
            teamStats[wteam]['FF'].append(wpf)
            teamStats[wteam]['FA'].append(lpf)
        except KeyError:
            teamStats[wteam] = {'Opp': [lteam], 'PF': [wscore], 'PA': [lscore], 'FGMF':[wfgm], 'FGMA':[lfgm],
                                'FGAF': [wfga], 'FGAA': [lfga], 'FGM3F': [wfgm3], 'FGM3A':[lfgm3], 'FGA3F':[wfga],
                                'FGA3A': [lfga3], 'FTMF': [wftm], 'FTMA': [lftm], 'FTAF':[wfta], 'FTAA':[lfta],
                                'ORF': [wor], 'ORA': [lor], 'DRF': [wdr], 'DRA':[ldr], 'ASTF':[wast],
                                'ASTA': [last], 'TOF': [wto], 'TOA': [lto], 'BLKF':[wblk], 'BLKA':[lblk],
                                'STLF': [wstl], 'STLA': [lstl], 'FF': [wpf], 'FA':[lpf]}
        try:
            teamStats[lteam]['Opp'].append(wteam)
            teamStats[lteam]['PA'].append(wscore)
            teamStats[lteam]['PF'].append(lscore)
            teamStats[lteam]['FGMF'].append(lfgm)
            teamStats[lteam]['FGMA'].append(wfgm)
            teamStats[lteam]['FGAF'].append(lfga)
            teamStats[lteam]['FGAA'].append(wfga)
            teamStats[lteam]['FGM3F'].append(lfgm3)
            teamStats[lteam]['FGM3A'].append(wfgm3)
            teamStats[lteam]['FGA3F'].append(lfga3)
            teamStats[lteam]['FGA3A'].append(wfga3)
            teamStats[lteam]['FTMF'].append(lftm)
            teamStats[lteam]['FTMA'].append(wftm)
            teamStats[lteam]['FTAF'].append(lfta)
            teamStats[lteam]['FTAA'].append(wfta)
            teamStats[lteam]['ORF'].append(lor)
            teamStats[lteam]['ORA'].append(wor)
            teamStats[lteam]['DRF'].append(ldr)
            teamStats[lteam]['DRA'].append(wdr)
            teamStats[lteam]['ASTF'].append(last)
            teamStats[lteam]['ASTA'].append(wast)
            teamStats[lteam]['TOF'].append(lto)
            teamStats[lteam]['TOA'].append(wto)
            teamStats[lteam]['BLKF'].append(lblk)
            teamStats[lteam]['BLKA'].append(wblk)
            teamStats[lteam]['STLF'].append(lstl)
            teamStats[lteam]['STLA'].append(wstl)
            teamStats[lteam]['FF'].append(lpf)
            teamStats[lteam]['FA'].append(wpf)
        except KeyError:
            teamStats[lteam] = {'PA': [wscore], 'PF': [lscore], 'Opp': [wteam], 'FGMF':[lfgm], 'FGMA':[wfgm],
                                'FGAF': [lfga], 'FGAA': [wfga], 'FGM3F': [lfgm3], 'FGM3A':[wfgm3], 'FGA3F':[lfga],
                                'FGA3A': [wfga3], 'FTMF': [lftm], 'FTMA': [wftm], 'FTAF':[lfta], 'FTAA':[wfta],
                                'ORF': [lor], 'ORA': [wor], 'DRF': [ldr], 'DRA':[wdr], 'ASTF':[last],
                                'ASTA': [wast], 'TOF': [lto], 'TOA': [wto], 'BLKF':[lblk], 'BLKA':[wblk],
                                'STLF': [lstl], 'STLA': [wstl], 'FF': [lpf], 'FA':[wpf]}
    return teamStats

def adjust(teamStats, teams, eta = .01, max_iter = 25):

    # This function takes the previous function's outputs as inputs
    # Additionally, there is a convergence parameter eta and maximum iterations

    ## This is split into two sections
    ## Point differential reflects a team's ability on both ends of the court
    ## All other stats are one direction only which requires a tweak to methodology

    #######################################################
    ## Point Differential Only ############################
    #######################################################

    # Recursively update team ranks until the maximum change is reduced to eta
    max_error = 1
    while max_error > eta:
        errors = []
        team_updates = {}
        for team in teams:
            # For each team we need the avg points per game
            # average strength of victory, calculated using pythagorean expectation
            avgppg = np.mean([teamStats[team]['PF'][i] - teamStats[team]['PA'][i] for i in range(len(teamStats[team]['PF']))])
            num = [teamStats[team]['PF'][i]**13.91 for i in range(len(teamStats[team]['PF']))]
            den = [(teamStats[team]['PF'][i]**13.91 + teamStats[team]['PA'][i]**13.91) for i in range(len(teamStats[team]['PF']))]
            SOV = [2*num[i]/float(den[i]) - 1 for i in range(len(num))]
            avgSOV = np.mean(SOV)
            adjustments = []
            for g in range(len(teamStats[team]['PA'])):
                # For each game that team plays in a given season we calculate a game score G
                # This is a reflection of how well a team did relative to their average performance
                G = (2*teamStats[team]['PF'][g]**13.91)/float((teamStats[team]['PF'][g]**13.91+teamStats[team]['PA'][g]**13.91)) - 1
                G *= avgppg/float(avgSOV)
                # The adjustment is a reflection of the rank of the team played plus the game score
                adjustments.append(teams[teamStats[team]['Opp'][g]]['PTDIFF'] + G)
            # A team's updated rank is the average of these opponent rank + game scores over the season
            errors.append(abs(np.mean(adjustments) - teams[team]['PTDIFF']))
            team_updates[team] = np.mean(adjustments)
        for team in team_updates:
            teams[team]['PTDIFF'] = team_updates[team]
        max_error = np.max(errors)
        print max_error

    #######################################################
    ## All other stats ####################################
    #######################################################

    # The one direction adjusted statistics are a little trickier
    # The framework remains the same

    statNames = ['PF','FGMF','FGAF','FGM3F','FGA3F','FTMF','FTAF','ORF','DRF','ASTF','TOF','BLKF','STLF','FF']
    for stat in statNames:
        # Now we add the loop for each stat before looping through the teams
        max_error = 1
        # Both a teams adjusted stat as well as their adjusted stat alloed must be 
        # optimized simultaneously: stat and statA
        statA = stat[:-1] + 'A'
        print stat
        print statA
        allStat = []
        allStatA = []
        for team in teamStats:
            for i in range(len(teamStats[team][stat])):
                allStat.append(teamStats[team][stat][i])
                allStatA.append(teamStats[team][statA][i])
        # For pure stats we cannot compare a stat against a defensive complement
        # E.G. Point difference is point for - points againg
        # Threes made is absolute
        # Therefore we must compare to league average to decide what is a "positive" performance
        # Leaguewide averages and standard deviations are calculated below
        allAvgStat = np.mean(allStat)
        allAvgStatA = np.mean(allStatA)
        allStdStat = np.std(allStat)
        allStdStatA = np.std(allStatA)
        n = 0
        while max_error > eta and n < max_iter:
            n += 1
            team_updates = {stat:{},statA:{}}
            errors = []
            errorsA = []
            for team in teams:
                # For each team calculate average and std stat and statA
                # calculate the strength of stat(SOS) and statA (SOSA) 
                # strength is the cdf of the zscore spread from -1 to 1 
                avgStat = np.mean(teamStats[team][stat])
                stdStat = np.std(teamStats[team][stat])
                avgStatA = np.mean(teamStats[team][statA])
                stdStatA = np.std(teamStats[team][statA])
                SOS = st.norm.cdf((avgStat-allAvgStat)/float(allStdStat))*2.0 - 1.0
                SOSA = st.norm.cdf((allAvgStatA - avgStatA)/float(allStdStatA)) * 2.0 - 1.0
                adjustments = []
                adjustmentsA = []
                for g in range(len(teamStats[team][stat])):
                    # For each game that team plays in a given season we calculate a game score G
                    # This is a reflection of how well a team did relative to their average performance
                    # Calculate for stat and statA
                    G = st.norm.cdf((teamStats[team][stat][g] - allAvgStat)/float(allStdStat))* 2.0 - 1.0
                    G *= (avgStat-allAvgStat)/float(SOS)
                    GA = st.norm.cdf((allAvgStatA - teamStats[team][statA][g])/float(allStdStatA))* 2.0 - 1.0
                    GA *= (allAvgStatA-avgStatA)/float(SOSA)
                    adjustments.append(teams[teamStats[team]['Opp'][g]][statA] + G)
                    adjustmentsA.append(teams[teamStats[team]['Opp'][g]][stat] + GA)
                errors.append(abs(np.mean(adjustments) - teams[team][stat]))
                errorsA.append(abs(np.mean(adjustmentsA) - teams[team][statA]))
                team_updates[stat][team] = np.mean(adjustments)
                team_updates[statA][team] = np.mean(adjustmentsA)
            for team in teams:
                teams[team][stat] = team_updates[stat][team]
                teams[team][statA] = team_updates[statA][team]
            max_error = np.max([np.max(errors),np.max(errorsA)])
            print max_error

    return teams

if __name__ == '__main__':
    ## These are the required table from the kaggle competition
    reg = pd.read_csv('regular_season_detailed_results.csv', header = 0)
    tou = pd.read_csv('tourney_detailed_results.csv', header = 0)
    teams = pd.read_csv('teams.csv', header = 0)
    seasons = pd.read_csv('seasons.csv', header = 0)
    seeds = pd.read_csv('tourney_seeds.csv', header = 0)
    slots = pd.read_csv('tourney_slots.csv', header = 0)

    # The below code will return the opponent adjusted statistics for the 2002-2003 season

    reg2003 = reg.loc[reg.season == 2003,:]

    teams2003 = teamInit(reg2003,teams)
    teamStats2003 = teamStatsInit(reg2003)
    adjusted2003 = adjust(teamStats=teamStats2003,teams=teams2003,eta=.5)
    print adjusted2003[1112]

