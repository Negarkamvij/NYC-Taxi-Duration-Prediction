# January taxi and candidate-weather audit

## Findings
- All **6,405,008 January source rows** are preserved.
- The preliminary positive-duration, intended-month eligibility mask includes **6,400,390 rows**. The other **4,618 rows** remain in the source/audit table; they are not deleted.
- The preliminary validation boundary is **January 26, 2020, at 01:52:21 New York time**. The audit marks **2,096 training trips that cross this boundary**, leaving **5,118,216 preliminary training rows** and **1,280,078 validation rows**. These are audit views, not a finalized model-training population.
- All eligible pickups have a matching candidate weather timestamp. Matching coverage does not prove weather accuracy or real-time availability.
- The January candidate has **744 hourly timestamps**, with positive rain in **67 hours** and positive snowfall in **17 hours**. Four mixed hours count in both positive-rain and positive-snow totals.
- The validation period has **4 rain-only hours** and **no snow or mixed hours**. Its **50,922 rain-associated taxi rows** do not represent 50,922 independent weather events.
- The split crosses a weather hour, so one weather timestamp is shared across the two audit views. The final split should keep weather-hour groups together as well as purge rides crossing its boundary. Consecutive hours may also belong to the same weather event.

**Decision before training:** Do not claim that this January validation can measure snow performance or establish reliable rainy-weather performance. Review a broader modeling period and weather source before locking the split. If the project stays January-only, snow must remain exploratory and rainy-weather results must be described as limited.

**Source limitation:** ERA5 is a reconstruction of past weather. Matching its timestamps backward does not turn it into a historical live feed. The current weather work is a candidate audit, not a point-in-time production backtest.

No models have been trained. These are data-audit findings, not prediction results.

## Sources
- NYC TLC: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Open-Meteo historical weather documentation: https://open-meteo.com/en/docs/historical-weather-api
- ERA5 via Open-Meteo, Wall Street-area request. Source request and returned location are recorded in the notebook and provenance.
