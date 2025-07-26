import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
from datetime import datetime

class DistillationColumnSimulator:
    # Engineered simulation for the Benzene-Toluene distillation column.
    

    def __init__(self):
        
        self.MW_benzene = 78
        self.MW_toluene = 92
        self.BP_benzene_K = 353.3
        self.BP_toluene_K = 383.8
        self.latent_heat_kJ_kmol = 30000  # From 30 MJ/kmol

        # Feed Conditions
        self.F_mass_kgs = 2.5
        self.wF_benzene = 0.40  # weight fraction
        self.T_feed_K = 295
        self.Cp_feed_kJ_kgK = 1.84

        # Column Specifications
        self.wD_benzene = 0.97  # weight fraction in distillate
        self.wB_toluene = 0.98  # weight fraction in bottoms
        self.R = 3.5  # Reflux ratio
        self.plate_spacing_m = 0.46

        # VLE Data for Benzene-Toluene at 1 atm
        self.vle_data = {
            'x': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'y': [0, 0.22, 0.38, 0.51, 0.63, 0.7, 0.78, 0.85, 0.91, 0.96, 1.0]
        }
        
        
        self.assumed_tray_efficiency = 0.75 # Typical value
        self.CE_index = 800 # Cost index for estimation

        # Results storage
        self.results = {}

    def _convert_weight_to_mole_fractions(self):
        
        self.xF = (self.wF_benzene / self.MW_benzene) / \
                  ((self.wF_benzene / self.MW_benzene) + ((1 - self.wF_benzene) / self.MW_toluene))

        wB_benzene = 1 - self.wB_toluene
        self.xB = (wB_benzene / self.MW_benzene) / \
                  ((wB_benzene / self.MW_benzene) + (self.wB_toluene / self.MW_toluene))

        wD_toluene = 1 - self.wD_benzene
        self.xD = (self.wD_benzene / self.MW_benzene) / \
                  ((self.wD_benzene / self.MW_benzene) + (wD_toluene / self.MW_toluene))

        self.results['Feed Mole Fraction (Benzene)'] = self.xF
        self.results['Distillate Mole Fraction (Benzene)'] = self.xD
        self.results['Bottoms Mole Fraction (Benzene)'] = self.xB

    def _perform_material_balance(self):
        
        # Mass Balance
        self.D_mass_kgs = self.F_mass_kgs * (self.wF_benzene - (1 - self.wB_toluene)) / (self.wD_benzene - (1 - self.wB_toluene))
        self.B_mass_kgs = self.F_mass_kgs - self.D_mass_kgs
        self.results['Distillate Mass Flow (kg/s)'] = self.D_mass_kgs
        self.results['Bottoms Mass Flow (kg/s)'] = self.B_mass_kgs

        # Molar Balance
        self.MW_feed_avg = self.xF * self.MW_benzene + (1 - self.xF) * self.MW_toluene
        self.F_molar_kmol_hr = (self.F_mass_kgs * 3600) / self.MW_feed_avg
        
        self.D_molar_kmol_hr = self.F_molar_kmol_hr * (self.xF - self.xB) / (self.xD - self.xB)
        self.B_molar_kmol_hr = self.F_molar_kmol_hr - self.D_molar_kmol_hr
        self.results['Feed Molar Flow (kmol/hr)'] = self.F_molar_kmol_hr
        self.results['Distillate Molar Flow (kmol/hr)'] = self.D_molar_kmol_hr
        self.results['Bottoms Molar Flow (kmol/hr)'] = self.B_molar_kmol_hr

    def _calculate_feed_quality(self):
        
        T_boiling_avg_K = self.xF * self.BP_benzene_K + (1 - self.xF) * self.BP_toluene_K
        heat_to_vaporize_kJ_kg = self.Cp_feed_kJ_kgK * (T_boiling_avg_K - self.T_feed_K)
        latent_heat_kJ_kg = self.latent_heat_kJ_kmol / self.MW_feed_avg
        self.q = (latent_heat_kJ_kg + heat_to_vaporize_kJ_kg) / latent_heat_kJ_kg
        self.results['Feed Quality (q)'] = self.q

    def _perform_mccabe_thiele_analysis(self):
        
        x_eq, y_eq = self.vle_data['x'], self.vle_data['y']
        
        # Define operating line and q-line equations
        y_rect = lambda x: (self.R / (self.R + 1)) * x + self.xD / (self.R + 1)
        
        if self.q == 1:
            x_intersect = self.xF
        else:
            y_q = lambda x: (self.q / (self.q - 1)) * x - self.xF / (self.q - 1)
            x_intersect = fsolve(lambda x: y_rect(x) - y_q(x), self.xF)[0]
            
        L_strip = self.R * self.D_molar_kmol_hr + self.q * self.F_molar_kmol_hr
        V_strip = L_strip - self.B_molar_kmol_hr
        y_strip = lambda x: (L_strip / V_strip) * x - (self.B_molar_kmol_hr * self.xB) / V_strip

        # Step off the theoretical plates
        n_stages = 0
        y_step = self.xD
        x_step = self.xD
        
        # Rectifying section
        while x_step > self.xF:
            n_stages += 1
            x_step = np.interp(y_step, y_eq, x_eq)
            y_step = y_rect(x_step)
            if n_stages == 1: self.feed_tray = 1
            if x_step <= self.xF:
                self.feed_tray = n_stages

        # Stripping section
        while x_step > self.xB:
            y_step = y_strip(x_step)
            x_step = np.interp(y_step, y_eq, x_eq)
            n_stages += 1
            
        self.N_theoretical = n_stages
        self.N_actual = int(np.ceil(self.N_theoretical / self.assumed_tray_efficiency))
        
        self.results['Theoretical Plates (inc. Reboiler)'] = self.N_theoretical
        self.results['Feed Plate Location (from top)'] = self.feed_tray
        self.results['Assumed Tray Efficiency'] = self.assumed_tray_efficiency
        self.results['Actual Trays Required'] = self.N_actual

    def _calculate_energy_requirements(self):
        
        # Using boil-up rate from the problem statement for consistency
        boilup_ratio = 238.1 / 100
        self.V_bottom_kmol_hr = boilup_ratio * self.F_molar_kmol_hr
        self.Q_reboiler_kJ_hr = self.V_bottom_kmol_hr * self.latent_heat_kJ_kmol
        
        # Steam at 240 kPa has latent heat of approx. 2185 kJ/kg
        lambda_steam_kJ_kg = 2185
        self.steam_required_kg_hr = self.Q_reboiler_kJ_hr / lambda_steam_kJ_kg
        
        # Condenser duty
        V_top = self.D_molar_kmol_hr * (self.R + 1)
        self.Q_condenser_kJ_hr = V_top * self.latent_heat_kJ_kmol

        self.results['Boilup Rate (kmol/hr)'] = self.V_bottom_kmol_hr
        self.results['Reboiler Duty (kW)'] = self.Q_reboiler_kJ_hr / 3600
        self.results['Condenser Duty (kW)'] = self.Q_condenser_kJ_hr / 3600
        self.results['Steam Required at 240 kPa (kg/hr)'] = self.steam_required_kg_hr

    def _calculate_column_diameter(self):
        
        # Bottom of column (higher vapor load)
        V_bottom_m3_hr = (self.V_bottom_kmol_hr * 8.314 * self.BP_toluene_K) / 101.325
        
        # Case (d): Based on 1 m/s vapor velocity
        A_total_m2 = (V_bottom_m3_hr / 3600) / 1.0
        self.D_col_total_m = np.sqrt(4 * A_total_m2 / np.pi)
        
        # Case (e): Based on 0.75 m/s in free area
        # From notes: Free Area = 0.589 * d^2
        # V_bottom_m3_hr = (0.589 * d^2) * (0.75 * 3600)
        self.D_col_free_area_m = np.sqrt(V_bottom_m3_hr / (0.589 * 2700))
        
        self.results['Column Diameter (based on 1 m/s total area)'] = self.D_col_total_m
        self.results['Column Diameter (based on 0.75 m/s free area)'] = self.D_col_free_area_m
        self.H_col_m = (self.N_actual - 1) * self.plate_spacing_m + 2.0 # Add space top/bottom
        self.results['Estimated Column Height (m)'] = self.H_col_m

    def _perform_cost_estimation(self):
        
        # Capital Cost (CAPEX)
        shell_volume = np.pi * (self.D_col_free_area_m/2)**2 * self.H_col_m
        shell_cost = 1500 * shell_volume * (self.CE_index / 600)
        tray_cost = 400 * self.N_actual * (self.CE_index / 600)
        self.capital_cost = 4.5 * (shell_cost + tray_cost) # Installation factor
        
        # Operating Cost (OPEX)
        steam_cost_hr = (self.steam_required_kg_hr / 1000) * 25 # $/hr
        cooling_water_cost_hr = (self.Q_condenser_kJ_hr / 1e6) * 0.05 # $/hr
        self.annual_operating_cost = (steam_cost_hr + cooling_water_cost_hr) * 24 * 350
        
        self.results['Estimated Capital Cost ($)'] = self.capital_cost
        self.results['Annual Operating Cost ($)'] = self.annual_operating_cost
        self.results['Payback Period (years)'] = self.capital_cost / self.annual_operating_cost

    def plot_mccabe_thiele(self):
        
        x_eq, y_eq = self.vle_data['x'], self.vle_data['y']
        
        y_rect = lambda x: (self.R / (self.R + 1)) * x + self.xD / (self.R + 1)
        
        if self.q == 1:
            x_intersect = self.xF
        else:
            y_q = lambda x: (self.q / (self.q - 1)) * x - self.xF / (self.q - 1)
            x_intersect = fsolve(lambda x: y_rect(x) - y_q(x), self.xF)[0]
        y_intersect = y_rect(x_intersect)
            
        L_strip = self.R * self.D_molar_kmol_hr + self.q * self.F_molar_kmol_hr
        V_strip = L_strip - self.B_molar_kmol_hr
        y_strip = lambda x: (L_strip / V_strip) * x - (self.B_molar_kmol_hr * self.xB) / V_strip

        plt.figure(figsize=(10, 8))
        plt.plot(x_eq, y_eq, 'b-', label='Equilibrium Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='45° Line')
        
        plt.plot([self.xD, x_intersect], [self.xD, y_intersect], 'r-', label='Rectifying Line')
        plt.plot([self.xB, x_intersect], [self.xB, y_intersect], 'g-', label='Stripping Line')
        plt.plot([self.xF, x_intersect], [self.xF, y_intersect], 'm-', label='q-Line')
        
        plt.title('McCabe-Thiele Diagram for Benzene-Toluene System', fontsize=16)
        plt.xlabel('Liquid Mole Fraction of Benzene (x)', fontsize=12)
        plt.ylabel('Vapor Mole Fraction of Benzene (y)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_temperature_profile(self):
       
        stages = np.arange(self.N_actual + 2) # Condenser, trays, reboiler
        temps = np.linspace(self.BP_benzene_K, self.BP_toluene_K, len(stages))
        
        plt.figure(figsize=(8, 6))
        plt.plot(temps - 273.15, stages, 'b-o')
        plt.ylabel('Stage Number (0=Condenser)')
        plt.xlabel('Temperature (°C)')
        plt.title('Column Temperature Profile')
        plt.gca().invert_yaxis() # Put condenser at the top
        plt.grid(True)
        plt.show()

    def export_to_excel(self, filename="Distillation_Design_Report.xlsx"):
        
        df = pd.DataFrame.from_dict(self.results, orient='index', columns=['Value'])
        df.index.name = "Parameter"
        
        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, sheet_name='Design_Summary')
        print(f"\nDesign report successfully exported to {filename}")

    def run_full_design(self):
        
        self._convert_weight_to_mole_fractions()
        self._perform_material_balance()
        self._calculate_feed_quality()
        self._perform_mccabe_thiele_analysis()
        self._calculate_energy_requirements()
        self._calculate_column_diameter()
        self._perform_cost_estimation()
        
        # Print summary to console
        print("Distillation Column Design")
        for key, value in self.results.items():
            if isinstance(value, (float, np.floating)):
                print(f"{key:45s}: {value:.4f}")
            else:
                print(f"{key:45s}: {value}")
       
        
       
        self.plot_mccabe_thiele()
        self.plot_temperature_profile()
        self.export_to_excel()

if __name__ == '__main__':
    design_simulator = DistillationColumnSimulator()
    design_simulator.run_full_design()