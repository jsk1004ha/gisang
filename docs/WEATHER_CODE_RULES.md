# Weather Code Rules

Initial weather state is rule-based. It avoids claiming a learned weather classifier before precipitation/cloud/visibility data are stable.

Priority:
1. Heavy precipitation: `heavy_rain` or `snow` when cold.
2. Light precipitation: `rain` or `snow` when cold.
3. Fog: high humidity and temperature close to dew point.
4. Strong wind.
5. Cloud cover: clear / mostly_clear / partly_cloudy / cloudy / overcast.
6. Heat/cold extremes.
7. `unknown` when key inputs are missing.
