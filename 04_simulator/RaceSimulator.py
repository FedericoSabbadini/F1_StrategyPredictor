import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modelQuery import RaceSimulator
from modelLoad import Race
import base64, io


def simulate(yearI: int, roundI: int, driverI: str):
    fig = None
    
    YEAR = yearI
    ROUND = roundI
    DRIVER = driverI

    # Compound colors and labels
    COMPOUND_COLORS = {
        'SOFT': '#FF3333',
        'MEDIUM': '#FFD700',
        'HARD': '#E8E8E8',
        'INTERMEDIATE': '#39FF14'
    }
    COMPOUND_SHORT = {'SOFT': 'S', 'MEDIUM': 'M', 'HARD': 'H', 'INTERMEDIATE': 'I'}

    race = Race()

    # Config
    df_f1 = race.df_f1
    SEQ_LEN = race.seq_len
    FEATURES = race.features
    PIT_THRESHOLD = race.pit_threshold # best simulation accuracy

    race_data = df_f1[(df_f1['Year'] == YEAR) & (df_f1['Round'] == ROUND)].copy()
    race_name = race_data['RaceName'].iloc[0] if len(race_data) > 0 else None

    # Run simulation
    simulator = RaceSimulator(race)

    sim_df, pit_laps, sc_laps, recommended_pits, num_stints, THRS = simulator.simulate(race_data, DRIVER)


    fig, axes = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1])
    data = sim_df[sim_df['DataAvailable']]
    total_laps = int(sim_df['TotalLaps'].iloc[0])
    STAR_SIZE = 200
    pit_laps_sorted = sorted(list(pit_laps))

    recommended_pits_sorted = sorted(list(recommended_pits.values()), key=lambda x: x['lap'])
    recommended_laps = [rec['lap'] for rec in recommended_pits_sorted]

    # ====== TOP PLOT: Pit Probability ======
    ax1 = axes[0]

    # Extend P(Pit) curve to start from lap 0
    laps = [0] + data['Lap'].tolist()
    probs = [0] + data['PitProb'].tolist()

    # P(Pit) curve
    ax1.plot(laps, probs, 'b-', linewidth=2, label='P(Pit)')
    ax1.fill_between(laps, 0, probs, alpha=0.2, color='blue')

    # Threshold line
    ax1.axhline(y=THRS, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'Threshold ({THRS:.0%})')

    # Actual pit stops (stars)
    for i, pit in enumerate(pit_laps_sorted):
        # Interpolate probability at pit lap
        pit = pit + 1
        if pit < data['Lap'].min() or pit > data['Lap'].max():
            prob = THRS
        else:
            prob = np.interp(pit, data['Lap'].values, data['PitProb'].values)

        ax1.scatter([pit], [prob], color='gold', s=STAR_SIZE, marker='*',
                    zorder=6, edgecolors='black', linewidths=1,
                    label='Real Pit' if i == 0 else '')

    # Predicted pit stops (green lines)
    for i, info in enumerate(recommended_pits.values()):
        ax1.axvline(x=info['lap'], color='green', linewidth=2, alpha=0.7,
                    label='Predicted Pit' if i == 0 else '')

    ax1.set_xlim(0, total_laps)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('P(Pit)')
    ax1.set_title(f'{DRIVER} - {race_name} {YEAR}')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ====== BOTTOM PLOT: Tire Strategy ======
    ax2 = axes[1]
    sim_df_sorted = sim_df.sort_values(['Stint', 'Lap'])

    # Actual strategy (solid lines)
    for stint_int, sdata in sim_df_sorted.groupby('Stint'):
        compound = sdata['Compound'].iloc[0]
        color = COMPOUND_COLORS.get(compound, 'gray')
        min_lap = max(laps)
        laps = sdata['Lap'].tolist()

        # Original tyre age for actual data
        tyre_age = list(range(len(laps)))
        # For first stint, fill missing laps from 0 to min_lap - 1
        if stint_int == 1:
            min_lap = laps[0]
            max_lap = laps[-1]

            # Extend backwards to lap 0 if needed
            if min_lap > 0:
                laps = list(range(0, min_lap)) + laps

            # Include last lap + 1
            laps = laps + [max_lap + 1]

            # Normalize to fill any missing laps
            lap_start = laps[0]
            lap_end = laps[-1]
            laps = list(range(lap_start, lap_end + 1))

            # Continuous tyre age
            tyre_age = [lap - lap_start for lap in laps]

        else:
            # Other stints, add extra laps before stint for plotting continuity
            max_lap = max(laps) + 1
            for i in range(min_lap, max_lap + 1):
                if i not in laps:
                    laps.append(i)
            laps = sorted(laps)
            tyre_age = [lap - laps[0] for lap in laps]

        ax2.plot(laps, tyre_age, color=color, linewidth=4, zorder=2)


    # Build predicted compounds dict
    pred_compounds = {}
    for pit_lap in recommended_laps:
        pit_row = sim_df[sim_df['Lap'] == pit_lap]
        next_stint = int(pit_row['Stint'].iloc[0]) + 1
        pred_compound = pit_row['CompPred'].iloc[0]
        pred_compounds[next_stint] = pred_compound

    # Predicted strategy (dashed lines)
    pred_pits_sorted = []
    for stint, info in recommended_pits.items():
        next_stint = stint + 1
        if next_stint in pred_compounds:
            pred_pits_sorted.append((info['lap'], pred_compounds[next_stint]))

    pred_pits_sorted = sorted(pred_pits_sorted, key=lambda x: x[0])
    for i, (pred_lap, pred_compound) in enumerate(pred_pits_sorted):
        next_lap = pred_pits_sorted[i + 1][0] if i + 1 < len(pred_pits_sorted) else total_laps
        pred_laps = list(range(pred_lap, next_lap + 1))
        pred_tyre_age = list(range(len(pred_laps)))
        color = COMPOUND_COLORS.get(pred_compound, 'gray')
        ax2.plot(pred_laps, pred_tyre_age, color=color, linewidth=2,
                linestyle='--', zorder=3)
    # Pit markers
    STAR_Y = 1
    for pit in pit_laps_sorted:
        pit = pit + 1
        ax2.scatter(pit, STAR_Y, color='gold', s=STAR_SIZE, marker='*',
                    zorder=6, edgecolors='black', linewidths=1)

    for info in recommended_pits.values():
        ax2.axvline(x=info['lap'], color='green', linestyle='--',
                    linewidth=1.5, alpha=0.5)

    # Legend
    ax2.plot([], [], color='gray', linewidth=4, label='Real')
    ax2.plot([], [], color='gray', linewidth=2, linestyle='--', label='Predicted')

    ax2.set_xlim(0, total_laps)
    ax2.set_ylim(0, None)
    ax2.set_xlabel('Lap')
    ax2.set_ylabel('Tire Age')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    simulation_file = sim_df.to_csv(index=False)

    # --- summary ---
    pit_summary = []
    for stint in range(1, num_stints + 1):
        row = {
            'Stint': stint,
            'Compound': sim_df[sim_df['Stint'] == stint]['Compound'].iloc[0]
        }

        if stint == num_stints:
            row.update({
                'RecLap': None, 'RecProb': None, 'ActualLap': None,
                'Diff': None, 'NextCompound': None, 'PredCompound': None,
                'CompMatch': None
            })
        elif stint in recommended_pits:
            rec = recommended_pits[stint]
            row['RecLap'] = rec['lap']
            row['RecProb'] = round(rec['prob'], 3)

            pit_rows = sim_df[
                (sim_df['Stint'] == stint + 1) &
                (sim_df['PitThisLap'] == 1)
            ]

            if len(pit_rows) > 0:
                actual = int(pit_rows['Lap'].iloc[0])
                row['ActualLap'] = actual
                row['Diff'] = actual - rec['lap']
                row['NextCompound'] = pit_rows['CompActual'].iloc[0]
                row['PredCompound'] = pit_rows['CompPred'].iloc[0]
                row['CompMatch'] = row['NextCompound'] == row['PredCompound']

        pit_summary.append(row)

    pitSummary_file = pd.DataFrame(pit_summary).to_csv(index=False)

    # --- chart ---
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    plt.close("all")

    return simulation_file, pitSummary_file, chart_base64
