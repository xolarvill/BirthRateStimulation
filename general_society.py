import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class PopulationSimulator:
    def __init__(self, 
                 initial_population=10000, 
                 initial_age_distribution=None,
                 max_age=100,
                 fertility_rates=None,
                 mortality_rates=None,
                 simulation_years=50):
        """
        初始化人口模拟器
        
        参数:
        - initial_population: 初始总人口数
        - initial_age_distribution: 初始年龄分布 (概率分布，总和为1)
        - max_age: 最大年龄
        - fertility_rates: 各年龄段的生育率 (按年龄索引的数组)
        - mortality_rates: 各年龄段的死亡率 (按年龄索引的数组)
        - simulation_years: 模拟年数
        """
        self.max_age = max_age
        self.simulation_years = simulation_years
        
        # 设置默认的初始年龄分布 (如果未提供)
        if initial_age_distribution is None:
            # 创建一个略微偏向年轻人口的分布
            x = np.arange(max_age)
            initial_age_distribution = np.exp(-0.03 * x)
            initial_age_distribution = initial_age_distribution / np.sum(initial_age_distribution)
        
        # 设置默认的生育率 (如果未提供)
        if fertility_rates is None:
            # 默认生育率: 15-49岁女性可生育，高峰在25-35岁
            fertility_rates = np.zeros(max_age)
            fertility_rates[15:50] = 0.08 * np.exp(-0.1 * np.abs(np.arange(15, 50) - 30))
        
        # 设置默认的死亡率 (如果未提供)
        if mortality_rates is None:
            # 简化的死亡率: 婴儿期略高，然后在中年缓慢上升，老年急剧上升
            mortality_rates = np.zeros(max_age)
            mortality_rates[0] = 0.01  # 婴儿死亡率
            mortality_rates[1:60] = 0.001 + 0.0001 * np.arange(1, 60)  # 缓慢上升
            mortality_rates[60:] = 0.015 * np.exp(0.08 * np.arange(0, max_age-60))  # 老年急剧上升
            mortality_rates = np.minimum(mortality_rates, 1.0)  # 确保概率不超过1
            
        # 初始化人口
        self.population = np.round(initial_population * initial_age_distribution).astype(int)
        self.fertility_rates = fertility_rates
        self.mortality_rates = mortality_rates
        
        # 储存历史数据
        self.history = {
            'total_population': [],
            'age_distribution': [],
            'births': [],
            'deaths': [],
            'dependency_ratio': [],
            'median_age': []
        }
        
        # 记录初始状态
        self._record_state()
        
    def set_fertility_scenario(self, scenario, intensity=1.0):
        """
        设置生育率情景
        
        参数:
        - scenario: 情景名称 ('high', 'medium', 'low', 'custom')
        - intensity: 情景强度调整因子
        """
        base_rates = np.zeros(self.max_age)
        
        if scenario == 'high':
            # 高生育率情景: 总和生育率约2.5
            base_rates[15:50] = 0.12 * np.exp(-0.1 * np.abs(np.arange(15, 50) - 28))
        elif scenario == 'medium':
            # 中等生育率情景: 总和生育率约2.1 (更替水平)
            base_rates[15:50] = 0.08 * np.exp(-0.1 * np.abs(np.arange(15, 50) - 30))
        elif scenario == 'low':
            # 低生育率情景: 总和生育率约1.5
            base_rates[15:50] = 0.055 * np.exp(-0.1 * np.abs(np.arange(15, 50) - 32))
        elif scenario == 'very_low':
            # 极低生育率情景: 总和生育率约1.2
            base_rates[15:50] = 0.04 * np.exp(-0.1 * np.abs(np.arange(15, 50) - 34))
        elif scenario == 'custom':
            # 保持当前设置
            return
        
        # 应用强度调整
        self.fertility_rates = base_rates * intensity
    
    def simulate_year(self):
        """模拟一年的人口变化"""
        # 计算新生儿
        female_ratio = 0.49  # 假设49%的人口为女性
        births = np.sum(self.population * self.fertility_rates * female_ratio)
        births = int(np.round(births))
        
        # 计算死亡人数
        deaths_by_age = np.random.binomial(self.population, self.mortality_rates)
        total_deaths = np.sum(deaths_by_age)
        
        # 更新人口年龄结构
        new_population = np.zeros(self.max_age)
        new_population[0] = births  # 新生儿
        new_population[1:] = self.population[:-1] - deaths_by_age[:-1]  # 其他人口老一岁，减去死亡
        
        # 更新当前人口
        self.population = np.maximum(0, new_population).astype(int)
        
        # 记录本年度状态
        self._record_state()
        
        return births, total_deaths
    
    def simulate(self):
        """模拟整个时间段"""
        for _ in range(self.simulation_years):
            self.simulate_year()
        
        return self.get_history()
    
    def _record_state(self):
        """记录当前状态到历史数据"""
        total_pop = np.sum(self.population)
        if total_pop > 0:  # 防止除以零
            age_dist = self.population / total_pop
            young_dependency = np.sum(self.population[:15]) / np.maximum(1, np.sum(self.population[15:65]))
            old_dependency = np.sum(self.population[65:]) / np.maximum(1, np.sum(self.population[15:65]))
            dependency_ratio = young_dependency + old_dependency
            
            # 计算中位年龄
            cumulative = np.cumsum(self.population)
            median_idx = np.searchsorted(cumulative, total_pop / 2)
            median_age = min(median_idx, self.max_age - 1)
        else:
            age_dist = np.zeros(self.max_age)
            dependency_ratio = 0
            median_age = 0
        
        # 更新历史数据
        self.history['total_population'].append(total_pop)
        self.history['age_distribution'].append(age_dist)
        self.history['births'].append(np.sum(self.population * self.fertility_rates * 0.49))
        self.history['deaths'].append(np.sum(self.population * self.mortality_rates))
        self.history['dependency_ratio'].append(dependency_ratio)
        self.history['median_age'].append(median_age)
    
    def get_history(self):
        """获取历史数据作为DataFrame"""
        df = pd.DataFrame({
            'year': np.arange(len(self.history['total_population'])),
            'total_population': self.history['total_population'],
            'births': self.history['births'],
            'deaths': self.history['deaths'],
            'dependency_ratio': self.history['dependency_ratio'],
            'median_age': self.history['median_age']
        })
        return df
    
    def plot_population_pyramid(self, year_idx=-1, figsize=(10, 8)):
        """绘制特定年份的人口金字塔"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取数据
        population = self.population if year_idx == -1 else (self.history['age_distribution'][year_idx] * self.history['total_population'][year_idx]).astype(int)
        
        # 计算男女性别分布 (假设)
        male_pop = population * 0.51  # 假设51%为男性
        female_pop = population * 0.49  # 假设49%为女性
        
        # 年龄组和标签
        age_groups = np.arange(0, self.max_age, 5)
        age_labels = [f"{i}-{i+4}" for i in age_groups[:-1]]
        age_labels.append(f"{age_groups[-1]}+")
        
        # 计算每个年龄组的人口
        male_data = []
        female_data = []
        for i in range(len(age_groups)-1):
            start, end = age_groups[i], min(age_groups[i+1], self.max_age)
            male_data.append(np.sum(male_pop[start:end]))
            female_data.append(np.sum(female_pop[start:end]))
        
        # 最后一个年龄组
        male_data.append(np.sum(male_pop[age_groups[-1]:]))
        female_data.append(np.sum(female_pop[age_groups[-1]:]))
        
        # 创建横向条形图
        y_pos = np.arange(len(age_labels))
        ax.barh(y_pos, -np.array(male_data), color='blue', alpha=0.7, label='男性')
        ax.barh(y_pos, female_data, color='red', alpha=0.7, label='女性')
        
        # 设置图表标签和标题
        year = year_idx if year_idx != -1 else len(self.history['total_population']) - 1
        ax.set_title(f"Population Pyramid (Year {year})")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(age_labels)
        ax.set_ylabel("Age groups")
        ax.set_xlabel("Population")
        
        # 处理 x 轴标签以显示正值
        ticks = ax.get_xticks()
        ax.set_xticklabels([str(abs(int(tick))) for tick in ticks])
        
        # 添加图例
        ax.legend()
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_demographic_trends(self, figsize=(15, 10)):
        """绘制人口统计趋势图"""
        history_df = self.get_history()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 总人口趋势
        axes[0, 0].plot(history_df['year'], history_df['total_population'], 'b-', linewidth=2)
        axes[0, 0].set_title('Whole population change')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Population')
        axes[0, 0].grid(True)
        
        # 出生和死亡率
        axes[0, 1].plot(history_df['year'], history_df['births'], 'g-', label='Births')
        axes[0, 1].plot(history_df['year'], history_df['deaths'], 'r-', label='Deaths')
        axes[0, 1].set_title('Births and deaths')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Number')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 中位年龄
        axes[1, 0].plot(history_df['year'], history_df['median_age'], 'purple', linewidth=2)
        axes[1, 0].set_title('Population median age')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Age')
        axes[1, 0].grid(True)
        
        # 抚养比
        axes[1, 1].plot(history_df['year'], history_df['dependency_ratio'], 'orange', linewidth=2)
        axes[1, 1].set_title('Ratio of rearing (0-14-year0olds and 65+year-olds to 15-64-year-olds)')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig, axes
    
    def animate_population_pyramid(self, interval=200):
        """创建人口金字塔的动画"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 初始化空金字塔
        age_groups = np.arange(0, self.max_age, 5)
        age_labels = [f"{i}-{i+4}" for i in age_groups[:-1]]
        age_labels.append(f"{age_groups[-1]}+")
        y_pos = np.arange(len(age_labels))
        
        male_bars = ax.barh(y_pos, np.zeros(len(age_labels)), color='blue', alpha=0.7, label='男性')
        female_bars = ax.barh(y_pos, np.zeros(len(age_labels)), color='red', alpha=0.7, label='女性')
        
        title = ax.set_title("Population pyramid (year: 0)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(age_labels)
        ax.set_ylabel("Age groups")
        ax.set_xlabel("Population")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        def init():
            for bar in male_bars:
                bar.set_width(0)
            for bar in female_bars:
                bar.set_width(0)
            return male_bars + female_bars + [title]
        
        def update(frame):
            # 获取该年份的人口分布
            if frame < len(self.history['age_distribution']):
                population = self.history['age_distribution'][frame] * self.history['total_population'][frame]
                
                # 计算男女性别分布
                male_pop = population * 0.51
                female_pop = population * 0.49
                
                # 计算每个年龄组的人口
                male_data = []
                female_data = []
                for i in range(len(age_groups)-1):
                    start, end = age_groups[i], min(age_groups[i+1], self.max_age)
                    male_data.append(np.sum(male_pop[start:end]))
                    female_data.append(np.sum(female_pop[start:end]))
                
                # 最后一个年龄组
                male_data.append(np.sum(male_pop[age_groups[-1]:]))
                female_data.append(np.sum(female_pop[age_groups[-1]:]))
                
                # 更新条形图
                for i, bar in enumerate(male_bars):
                    bar.set_width(-male_data[i])
                for i, bar in enumerate(female_bars):
                    bar.set_width(female_data[i])
                
                # 更新x轴范围
                max_val = max(max(male_data), max(female_data)) * 1.1
                ax.set_xlim(-max_val, max_val)
                
                # 处理x轴标签以显示正值
                ticks = np.linspace(-max_val, max_val, 9)
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(abs(int(tick))) for tick in ticks])
                
                # 更新标题
                title.set_text(f"Population pyramid (year: {frame})")
            
            return male_bars + female_bars + [title]
        
        ani = FuncAnimation(fig, update, frames=range(len(self.history['age_distribution'])),
                            init_func=init, blit=False, interval=interval)
        
        plt.tight_layout()
        return ani
        
    def compare_scenarios(scenarios_dict, simulation_years=50, initial_population=10000, figsize=(15, 12)):
        """
        比较不同生育率情景下的人口结构变化
        
        参数:
        - scenarios_dict: 字典，键为情景名称，值为生育率情景设置 ('high', 'medium', 'low', 'very_low')
        - simulation_years: 模拟年数
        - initial_population: 初始人口数量
        - figsize: 图表大小
        
        返回:
        - 比较图表和各情景的模拟器对象
        """
        # 存储各情景的模拟器和结果
        simulators = {}
        results = {}
        
        # 对每个情景运行模拟
        for name, scenario in scenarios_dict.items():
            simulator = PopulationSimulator(initial_population=initial_population, 
                                           simulation_years=simulation_years)
            
            if isinstance(scenario, tuple) and len(scenario) == 2:
                # 如果提供的是(情景名称，强度)元组
                simulator.set_fertility_scenario(scenario[0], scenario[1])
            else:
                # 如果只提供情景名称
                simulator.set_fertility_scenario(scenario)
                
            simulator.simulate()
            simulators[name] = simulator
            results[name] = simulator.get_history()
        
        # 创建比较图表
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios_dict)))
        
        # 总人口趋势
        for i, (name, df) in enumerate(results.items()):
            axes[0, 0].plot(df['year'], df['total_population'], color=colors[i], 
                           linewidth=2, label=name)
        axes[0, 0].set_title('Total population change in different scenarios')
        axes[0, 0].set_xlabel('year')
        axes[0, 0].set_ylabel('population')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 出生数
        for i, (name, df) in enumerate(results.items()):
            axes[0, 1].plot(df['year'], df['births'], color=colors[i], 
                           linewidth=2, label=name)
        axes[0, 1].set_title('births in different scenarios')
        axes[0, 1].set_xlabel('year')
        axes[0, 1].set_ylabel('births')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 中位年龄
        for i, (name, df) in enumerate(results.items()):
            axes[1, 0].plot(df['year'], df['median_age'], color=colors[i], 
                           linewidth=2, label=name)
        axes[1, 0].set_title('population median age in differents scenarios')
        axes[1, 0].set_xlabel('year')
        axes[1, 0].set_ylabel('age')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 抚养比
        for i, (name, df) in enumerate(results.items()):
            axes[1, 1].plot(df['year'], df['dependency_ratio'], color=colors[i], 
                           linewidth=2, label=name)
        axes[1, 1].set_title('ratios of bearing in different scenarios')
        axes[1, 1].set_xlabel('year')
        axes[1, 1].set_ylabel('ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        return fig, axes, simulators


# 使用示例
if __name__ == "__main__":
    society = 'general'
    # 示例1: 单一情景模拟
    print("运行单一情景模拟...")
    sim = PopulationSimulator(initial_population=10000, simulation_years=50)
    sim.set_fertility_scenario('very_low')  # 设置极低生育率情景
    sim.simulate()
    
    # 绘制人口金字塔
    fig1, _ = sim.plot_population_pyramid()
    plt.savefig(f'{society}_population_pyramid.png')
    
    # 绘制人口趋势
    fig2, _ = sim.plot_demographic_trends()
    plt.savefig(f'{society}_demographic_trends.png')
    
    # 示例2: 多情景比较
    print("运行多情景比较...")
    scenarios = {
        '高生育率': 'high',
        '中等生育率': 'medium',
        '低生育率': 'low',
        '极低生育率': 'very_low'
    }
    
    fig3, _, _ = PopulationSimulator.compare_scenarios(scenarios, simulation_years=50)
    plt.savefig(f'{society}_scenario_comparison.png')
    
    print("模拟完成，图表已保存。")