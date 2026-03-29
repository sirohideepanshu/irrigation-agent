class SmartIrrigationAgent:
    def __init__(self):
        self.prev_soil = {}

    def predict_soil(self, soil, water, rain, temp):
        water_effect = 0.8 * water
        rain_effect = 4 if rain == 1 else 0
        evaporation = max(0, (temp - 25) * 0.25)

        return soil + water_effect + rain_effect - evaporation

    def get_action(self, state, zone_id):
        soil = state["soil_moisture"][zone_id]
        prev = self.prev_soil.get(zone_id, soil)

        delta = soil - prev

        rain = int(state.get("rain_forecast", 0))
        temp = float(state.get("temperature", 25))

        TARGET = 52
        LOW = 46
        HIGH = 58

        error = TARGET - soil
        water = 0.5 * error

        if soil < 25:
            water = 8
        elif soil < 35:
            water += 3
        elif soil < 45:
            water += 1

        if rain == 1:
            water *= 0.7
            if soil > 52:
                water -= 1

        if temp > 35:
            water += 2

        predicted = self.predict_soil(soil, water, rain, temp)

        if predicted > HIGH:
            water -= 2
        elif predicted < LOW:
            water += 2

        if delta > 5:
            water -= 2

        if soil > 65:
            water = 0

        if soil < 40 and water < 1:
            water = 1

        water = max(0, min(12, int(round(water))))

        self.prev_soil[zone_id] = soil

        return {
            "zone_id": zone_id,
            "water_mm": water
        }