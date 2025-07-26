import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
import pandas as pd
from datetime import datetime

class BinaryDistillationDesign:
    """
    Advanced binary distillation column design system.
    Handles rigorous calculations for two-component systems only.
    
    Parameters:
    -----------
    light_component : str
        Name of light component
    heavy_component : str
        Name of heavy component
    feed_comp : float
        Mole fraction of light component in feed (0-1)
    F : float
        Feed flow rate (kmol/hr)
    xD : float
        Mole fraction of light component in distillate (0-1)
    xB : float
        Mole fraction of light component in bottoms (0-1)
    P : float
        Operating pressure (kPa)
    q : float
        Feed quality (0-1)
    R_factor : float
        Multiplier for minimum reflux ratio
    tray_efficiency : float
        Overall tray efficiency (0-1)
    spacing : float
        Tray spacing (m)
    CE_index : float
        Current cost index for economic calculations
    """
    
    def __init__(self, light_component='ethanol', heavy_component='water',
                 feed_comp=0.4, F=100, xD=0.95, xB=0.05, P=101.325,
                 q=0.8, R_factor=1.5, tray_efficiency=0.7, spacing=0.5, CE_index=800):
        
        # System properties
        self.light_component = light_component
        self.heavy_component = heavy_component
        self.feed_comp = feed_comp  # mole fraction light component
        self.F = F  # kmol/hr
        self.xD = xD  # distillate composition
        self.xB = xB  # bottoms composition
        self.P = P  # kPa
        self.q = q  # feed quality
        self.R_factor = R_factor
        self.tray_efficiency = tray_efficiency
        self.spacing = spacing  # m
        self.CE_index = CE_index  # cost index
        
        # Physical properties (default for ethanol-water)
        self.rho_v = 1.2  # kg/m³ (vapor density)
        self.rho_l = 800  # kg/m³ (liquid density)
        self.MW_light = 46  # molecular weight (ethanol)
        self.MW_heavy = 18  # molecular weight (water)
        self.sigma = 0.022  # N/m (surface tension)
        self.lambda_light = 38500  # kJ/kmol (heat of vap, ethanol)
        self.lambda_heavy = 40650  # kJ/kmol (heat of vap, water)
        
        # Cost parameters ($)
        self.tray_cost = 350  # per tray
        self.shell_cost = 1200  # per m³
        self.steam_cost = 25  # per ton
        self.cooling_cost = 0.05  # per MJ
        self.installation_factor = 4.5
        
        # Results storage
        self.results = {}
        self.sensitivity_data = {}
        
    def material_balance(self) -> None:
        """Perform binary material balance calculations."""
        self.D = self.F * (self.feed_comp - self.xB) / (self.xD - self.xB)
        self.B = self.F - self.D
        
        self.results['Distillate Flow (kmol/hr)'] = self.D
        self.results['Bottoms Flow (kmol/hr)'] = self.B
        self.results['Distillate Composition'] = self.xD
        self.results['Bottoms Composition'] = self.xB
        
    def calculate_relative_volatility(self) -> float:
        """Calculate relative volatility for binary system."""
        # Using constant alpha for ethanol-water (could be replaced with VLE calculations)
        return 2.5
    
    def fug_method(self) -> None:
        """Fenske-Underwood-Gilliland method for binary systems."""
        alpha = self.calculate_relative_volatility()
        
        # Fenske equation for minimum stages
        self.N_min = np.log((self.xD/(1-self.xD)) * ((1-self.xB)/self.xB)) / np.log(alpha)
        
        # Underwood equations for minimum reflux
        def underwood_eq(theta):
            return (alpha*self.feed_comp)/(alpha-theta) + \
                   ((1-self.feed_comp))/(1-theta) - (1-self.q)
        
        theta = fsolve(underwood_eq, 1.5)[0]
        self.R_min = (alpha*self.xD)/(alpha-theta) + ((1-self.xD))/(1-theta) - 1
        self.R = self.R_factor * self.R_min
        
        # Gilliland correlation for actual stages
        X = (self.R - self.R_min)/(self.R + 1)
        Y = 1 - np.exp((1 + 54.4*X)/(11 + 117.2*X) * (X - 1)/np.sqrt(X))
        self.N_theoretical = int((Y + self.N_min)/(1 - Y))
        
        # Store results
        self.results['Minimum Stages'] = self.N_min
        self.results['Minimum Reflux Ratio'] = self.R_min
        self.results['Actual Reflux Ratio'] = self.R
        self.results['Theoretical Stages'] = self.N_theoretical
    
    def calculate_tray_efficiency(self) -> None:
        """Set actual trays based on constant tray efficiency."""
        # Using constant efficiency provided during initialization
        self.N_actual = int(np.ceil(self.N_theoretical / self.tray_efficiency))
        self.results['Tray Efficiency'] = self.tray_efficiency
        self.results['Actual Trays'] = self.N_actual
    
    def column_sizing(self) -> None:
        """Calculate column diameter and height."""
        V = self.D * (self.R + 1)  # vapor flow in rectifying section (kmol/hr)
        
        # Average molecular weight
        avg_MW = self.MW_light * self.xD + self.MW_heavy * (1-self.xD)
        
        # Set a typical operating velocity directly
        U_operating = 0.6 # m/s (typical vapor velocity)
        
        # Vapor volumetric flow rate (m³/s)
        Q_vapor = V * avg_MW / (self.rho_v * 3600)
        
        # Cross-sectional area and diameter
        A_cross = Q_vapor / U_operating
        self.D_col = np.sqrt(4 * A_cross / np.pi)
        self.H_col = self.N_actual * self.spacing
        
        # Store results
        self.results['Column Diameter (m)'] = self.D_col
        self.results['Column Height (m)'] = self.H_col
        self.results['Operating Velocity (m/s)'] = U_operating
    
    def energy_requirements(self) -> None:
        """Calculate condenser and reboiler duties for binary system."""
        V = self.D * (self.R + 1)  # kmol/hr
        
        # Average heat of vaporization (kJ/kmol)
        avg_lambda = self.lambda_light * self.xD + self.lambda_heavy * (1-self.xD)
        
        # Condenser duty (kJ/hr)
        self.Q_condenser = V * avg_lambda
        
        # Reboiler duty (kJ/hr)
        self.Q_reboiler = (V + self.B - self.F*self.q) * avg_lambda
        
        # Convert to kW and store
        self.results['Condenser Duty (kW)'] = self.Q_condenser / 3600
        self.results['Reboiler Duty (kW)'] = self.Q_reboiler / 3600
    
    def cost_estimation(self, project_life: int = 10) -> None:
        """Perform capital and operating cost estimation."""
        # Capital costs
        tray_cost = self.tray_cost * self.N_actual * (self.CE_index/600)
        shell_cost = self.shell_cost * np.pi*(self.D_col/2)**2 * self.H_col * (self.CE_index/600)
        total_capital = self.installation_factor * (tray_cost + shell_cost)
        
        # Operating costs
        steam_required = self.Q_reboiler / 2257  # tons/hr (using latent heat of water)
        cooling_required = self.Q_condenser / 3600  # MJ/hr
        
        annual_steam = steam_required * 24 * 350 * self.steam_cost
        annual_cooling = cooling_required * 24 * 350 * self.cooling_cost
        total_operating = annual_steam + annual_cooling
        
        # NPV calculation
        npv = -total_capital
        for year in range(1, project_life+1):
            npv += total_operating / (1.1)**year
        
        # Store results
        self.results['Capital Cost ($)'] = total_capital
        self.results['Annual Operating Cost ($)'] = total_operating
        self.results['10-year NPV ($)'] = npv
        self.results['Payback Period (years)'] = total_capital / total_operating
    
    def sensitivity_analysis(self, R_range: tuple = (1.2, 2.5, 10)) -> None:
        """
        Perform sensitivity analysis on reflux ratio factor.
        
        Parameters:
        -----------
        R_range : tuple
            (min, max, steps) for reflux ratio factor
        """
        R_factors = np.linspace(R_range[0], R_range[1], R_range[2])
        stages = []
        diameters = []
        costs = []
        
        original_R = self.R_factor
        
        for factor in R_factors:
            self.R_factor = factor
            self.fug_method()
            self.calculate_tray_efficiency()
            self.column_sizing()
            self.cost_estimation()
            
            stages.append(self.N_actual)
            diameters.append(self.D_col)
            costs.append(self.results['Capital Cost ($)'])
        
        # Restore original value
        self.R_factor = original_R
        
        # Store results
        self.sensitivity_data = {
            'R_factors': R_factors,
            'stages': stages,
            'diameters': diameters,
            'costs': costs
        }
        
        # Plot results
        self._plot_sensitivity(R_factors, stages, diameters, costs)
    
    def _plot_sensitivity(self, R_factors, stages, diameters, costs) -> None:
        """Plot sensitivity analysis results."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        ax1.plot(R_factors, stages, 'b-o')
        ax1.set_xlabel('Reflux Ratio Factor')
        ax1.set_ylabel('Number of Trays')
        ax1.grid(True)
        
        ax2.plot(R_factors, diameters, 'r-o')
        ax2.set_xlabel('Reflux Ratio Factor')
        ax2.set_ylabel('Column Diameter (m)')
        ax2.grid(True)
        
        ax3.plot(R_factors, costs, 'g-o')
        ax3.set_xlabel('Reflux Ratio Factor')
        ax3.set_ylabel('Capital Cost ($)')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.suptitle('Sensitivity Analysis: Reflux Ratio Factor', y=1.02)
        plt.show()
    
    def equilibrium_curve(self, x: np.ndarray) -> np.ndarray:
        """Calculate vapor-liquid equilibrium composition."""
        alpha = self.calculate_relative_volatility()
        return (alpha * x) / (1 + (alpha - 1) * x)
    
    def plot_mccabe_thiele(self) -> None:
        """Generate McCabe-Thiele diagram for binary system."""
        x = np.linspace(0, 1, 100)
        y = self.equilibrium_curve(x)

        # Define operating line and q-line equations
        y_rect = lambda x_val: (self.R / (self.R + 1)) * x_val + self.xD / (self.R + 1)
        
        L_strip = self.R * self.D + self.q * self.F
        V_strip = L_strip - self.B
        y_strip = lambda x_val: (L_strip / V_strip) * x_val - (self.B * self.xB) / V_strip
        
        # Find the intersection point of the operating lines
        if self.q == 1: # Handle vertical q-line for saturated liquid
            x_intersect = self.feed_comp
        else:
            y_q = lambda x_val: (self.q / (self.q - 1)) * x_val - self.feed_comp / (self.q - 1)
            # Solve for the intersection of the rectifying line and the q-line
            x_intersect = fsolve(lambda x_val: y_rect(x_val) - y_q(x_val), self.feed_comp)[0]
        
        y_intersect = y_rect(x_intersect)

        # Plotting
        plt.figure(figsize=(10, 8))
        plt.plot(x, y, 'b-', label='Equilibrium')
        plt.plot([0, 1], [0, 1], 'k--', label='45° line')

        # Plot lines between their correct intersection points
        plt.plot([self.xD, x_intersect], [self.xD, y_intersect], 'r-', label='Rectifying')
        plt.plot([self.xB, x_intersect], [self.xB, y_intersect], 'g-', label='Stripping')
        plt.plot([self.feed_comp, x_intersect], [self.feed_comp, y_intersect], 'm-', label='q-line')

        plt.xlabel('Liquid mole fraction ' + self.light_component)
        plt.ylabel('Vapor mole fraction ' + self.light_component)
        plt.title('McCabe-Thiele Diagram')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_temperature_profile(self) -> None:
        """Generate temperature profile through the column."""
        # Simplified linear profile (top to bottom)
        stages = np.arange(self.N_actual + 1) # Including reboiler
        temp_top = 78  # °C for ethanol-water (distillate)
        temp_bottom = 100  # °C for bottoms
        
        # Create linear temperature profile
        temp_profile = np.linspace(temp_top, temp_bottom, self.N_actual + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(stages, temp_profile, 'b-o')
        plt.xlabel('Stage Number (0=Condenser)')
        plt.ylabel('Temperature (°C)')
        plt.title('Column Temperature Profile')
        plt.grid(True)
        plt.show()
    
    def export_to_excel(self, filename: str = 'distillation_design.xlsx') -> None:
        """Export design results to Excel file."""
        df = pd.DataFrame.from_dict(self.results, orient='index', columns=['Value'])
        
        metadata = {
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Light Component': self.light_component,
            'Heavy Component': self.heavy_component,
            'Feed Composition': self.feed_comp,
            'Feed Flow (kmol/hr)': self.F,
            'Pressure (kPa)': self.P
        }
        meta_df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Design Parameters'])

        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, sheet_name='Results')
            meta_df.to_excel(writer, sheet_name='Metadata')
            if hasattr(self, 'sensitivity_data') and self.sensitivity_data:
                pd.DataFrame({
                    'Reflux Ratio Factor': self.sensitivity_data['R_factors'],
                    'Stages': self.sensitivity_data['stages'],
                    'Diameter (m)': self.sensitivity_data['diameters'],
                    'Cost ($)': self.sensitivity_data['costs']
                }).to_excel(writer, sheet_name='Sensitivity', index=False)
        
        print(f"Results exported to {filename}")
    
    def run_full_design(self) -> None:
        """Execute complete distillation column design workflow."""
        print(f"Running binary distillation design for {self.light_component}-{self.heavy_component} system...\n")
        
        # Design sequence
        self.material_balance()
        self.fug_method()
        self.calculate_tray_efficiency()
        self.column_sizing()
        self.energy_requirements()
        self.cost_estimation()
        
        # Visualization
        self.plot_mccabe_thiele()
        self.plot_temperature_profile()
        
        # Analysis
        self.sensitivity_analysis()
        self.export_to_excel()
        
        # Print summary
        print("\nDesign completed successfully!")
        print("="*50)
        for key, value in self.results.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:.2f}")
            else:
                print(f"{key:30s}: {value}")
        print("="*50)

if __name__ == "__main__":
    # Example usage
    column = BinaryDistillationDesign(
        light_component='ethanol',
        heavy_component='water',
        feed_comp=0.4,
        F=100,
        xD=0.95,
        xB=0.05,
        P=101.325,
        q=0.8,
        R_factor=1.5,
        tray_efficiency=0.75 # Setting a constant efficiency
    )
    
    # Run complete design
    column.run_full_design()