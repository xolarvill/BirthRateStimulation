from typing import List
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # 添加pandas导入


class People():
    def __init__(self):
        self.gender = None
        self.age = 0

class Society():
    def __init__(self, 
                 birth_rate: float,
                 initial_population: int, 
                 simulation_years: int,
                 scenario: str
                 ):
        
        self.initial_population = initial_population # 初始人口数量
        self.simulation_years = simulation_years # 模拟总年份
        self.birth_rate = birth_rate # 给定总出生率，其含义为每个可生育年龄女性的平均出生子女数量，是本模型考察的重点对象，而死亡率不是本模型的重点对象
        self.max_age = 100 # 最大年龄
        self.initial_year = 2022 # 初始年份
        self.scenario = scenario # 模拟的场景，支持reality, back to normal, early marriage三个模拟选项
        self.fertility_rates = self._get_fertility_rates()
        self.mortality_rates = self._get_mortality_rates()
        self.age_structure = self._get_age_structure() # 人口结构
        self.sex_ratio = 1.01 # 性别比，男性人口数量是女性人口数量的1.01倍
        self.births = 0
        self.deaths = 0
        self.population = []
        
    # 储存历史数据
        self.history = {
            'year_index': [],
            'total_population': [],
            'age_distribution': [],
            'births': [],
            'deaths': [],
            'dependency_ratio': [],
            'young_dependency_ratio': [],
            'elderly_dependency_ratio': [],
            'patriarchal_dependency_ratio': [],
            'median_age': [],
            'working_age_population': [],
            'elderly_population': [],
            'young_population': []
        }
        
        # 记录初始状态
        self._record_state()
        
    
    def _get_fertility_rates(self):
        """根据不同的生育率结构，返回不同年龄段的生育率"""
        fertility = np.zeros(self.max_age)
        if self.scenario == "reality":
            # 韩国当前由于其独特的社会原因，推崇晚婚晚育，生育年龄明显推迟
            # 韩国女性生育主要集中在25-39岁，但推迟生育趋势明显
            # 将0.78的总和生育率分配给不同年龄段
            fertility[20:25] = self.birth_rate * 0.05  # 20-24岁生育率很低
            fertility[25:30] = self.birth_rate * 0.10  # 25-29岁生育率开始上升
            fertility[30:35] = self.birth_rate * 0.40  # 30-34岁为生育高峰
            fertility[35:40] = self.birth_rate * 0.30  # 35-39岁仍有较高生育率
            fertility[40:45] = self.birth_rate * 0.10  # 40-44岁生育率降低
            fertility[45:50] = self.birth_rate * 0.05  # 45-49岁生育率很低
            return fertility
        
        elif self.scenario == 'back to normal':
            # 假设韩国社会回到一个正常生育结构，年轻女性生育积极度变高
            fertility[20:30] = self.birth_rate * 0.50  
            fertility[30:40] = self.birth_rate * 0.30  
            fertility[40:50] = self.birth_rate * 0.20  
            return fertility
        
        elif self.scenario == 'early marriage':
            # 假设韩国社会鼓励早婚，即女性更愿意在年轻的时候生育
            fertility[20:30] = self.birth_rate * 0.70  
            fertility[30:40] = self.birth_rate * 0.30  
            fertility[40:50] = self.birth_rate * 0.05 

            return fertility
    
    def _get_mortality_rates(self):
        mortality = np.zeros(self.max_age)
        # 韩国作为发达经济体拥有较低的死亡率
        # 同时中老年死亡率较高，这符合医学规律
        mortality[0] = 0.003 # 婴儿死亡率（非常低）
        mortality[1:15] = 0.0005 # 儿童及青少年死亡率（极低）
        mortality[15:40] = 0.001 # 青年死亡率
        mortality[40:65] = np.linspace(0.002, 0.01, 25) # 中年死亡率（逐渐上升）
        mortality[65:80] = np.linspace(0.015, 0.06, 15) # 老年死亡率（加速上升）
        mortality[80:90] = np.linspace(0.08, 0.2, 10)
        mortality[90:] = np.linspace(0.25, 0.5, self.max_age - 90)
        
        return mortality
    
    def _get_age_structure(self):
        # 模拟老龄化人口结构
        distribution = np.zeros(self.max_age)
        
        # 0-14岁：约12%
        distribution[0:15] = 0.12 / 15
        
        # 15-64岁：约71%，但呈现老龄化趋势
        # 分配更多权重给40-64岁人口
        distribution[15:40] = 0.33 / 25  # 年轻工作人口
        distribution[40:65] = 0.38 / 25  # 年长工作人口
        
        # 65岁以上：约17%
        # 韩国是快速老龄化的社会
        distribution[65:85] = 0.15 / 20
        distribution[85:] = 0.02 / 15
        
        # 归一化确保总和为1
        return distribution / np.sum(distribution)
        
    def initialize_population(self):
        # 根据给定的年龄分布self.age_structure、男女性别比self.sex_ratio和初始人口数量self.initial_population来创建初始人口
        for _ in range(self.initial_population):
            person = People()
            # 根据年龄分布，随机选择年龄
            person.age = np.random.choice(np.arange(self.max_age), p=self.age_structure)
            
            # 根据性别比，随机选择性别，生成一个在0和1之间的性别判断随机数，如果小于0.525则为男性，否则为女性
            if np.random.random() < 0.525:
                person.gender = "male"
            else:
                person.gender = "female"

            self.population.append(person)

    def _record_state(self):
        total_population = len(self.population)
        
        # 计算每个年龄段的人口数量，寻找年龄为0-15的people标记为young，记录其数量、性别比；15-65为working force，记录其数量、性别比；65岁以上为elderly，记录其数量、性别比
        young = [person for person in self.population if person.age < 15]
        young_male = [person for person in young if person.gender == "male"]
        young_female = [person for person in young if person.gender == "female"]
        n_young = len(young)
        n_young_male = len(young_male)
        n_young_female = len(young_female)
    
        working_age_prime = [person for person in self.population if person.age >= 15 and person.age <= 35]
        working_age_prime_male = [person for person in working_age_prime if person.gender == "male"]
        working_age_prime_female = [person for person in working_age_prime if person.gender == "female"]
        n_working_age_prime = len(working_age_prime)
        n_working_age_prime_male = len(working_age_prime_male)
        n_working_age_prime_female = len(working_age_prime_female)
    
        working_age_inferior = [person for person in self.population if person.age > 35 and person.age <= 64]
        working_age_inferior_male = [person for person in working_age_inferior if person.gender == "male"]
        working_age_inferior_female = [person for person in working_age_inferior if person.gender == "female"]
        n_working_age_inferior = len(working_age_inferior)
        n_working_age_inferior_male = len(working_age_inferior_male)
        n_working_age_inferior_female = len(working_age_inferior_female)
        
        working_age = working_age_prime + working_age_inferior
        n_working_age = n_working_age_prime + n_working_age_inferior
        n_working_age_male = n_working_age_prime_male + n_working_age_inferior_male
        n_working_age_female = n_working_age_prime_female + n_working_age_inferior_female
        
        elderly = [person for person in self.population if person.age >= 65]
        elderly_male = [person for person in elderly if person.gender == "male"]
        elderly_female = [person for person in elderly if person.gender == "female"]
        n_elderly = len(elderly)
        n_elderly_male = len(elderly_male)
        n_elderly_female = len(elderly_female)
        
        # 计算抚养率
        young_dependency_ratio = n_young / max(1, n_working_age)
        elderly_dependency_ratio = n_elderly / max(1, n_working_age)
        dependency_ratio = young_dependency_ratio + elderly_dependency_ratio
        
        # 计算男性抚养率
        patriarchal_dependency_ratio = (n_young + n_elderly + n_working_age_female) / max(1, n_working_age_male)
        
        # 计算中位数年龄
        median_age = np.median([person.age for person in self.population])
        
        # 计算年龄分布
        age_dist = np.zeros(self.max_age)
        for person in self.population:
            if person.age < self.max_age:
                age_dist[person.age] += 1
        if total_population > 0:
            age_dist = age_dist / total_population
        
        # 记录当前状态到历史数据self.history中
        self.history['year_index'].append(self.initial_year)
        self.history['total_population'].append(total_population)
        self.history['age_distribution'].append(age_dist)
        self.history['births'].append(self.births)
        self.history['deaths'].append(self.deaths)
        
        self.history['median_age'].append(median_age)
        
        self.history['dependency_ratio'].append(dependency_ratio)
        self.history['young_dependency_ratio'].append(young_dependency_ratio)
        self.history['elderly_dependency_ratio'].append(elderly_dependency_ratio)
        self.history['patriarchal_dependency_ratio'].append(patriarchal_dependency_ratio)
        
        self.history['working_age_population'].append(n_working_age)
        self.history['elderly_population'].append(n_elderly)
        self.history['young_population'].append(n_young)
    
    def simulate_one_year(self):
        self.initial_year += 1
        
        # 模拟一年内人口变化
        new_population = []
        old_population = self.population.copy()
    
        # 所有个体的年龄增长一岁
        for person in old_population:
            person.age += 1
        
        # 获取不同年龄段的死亡率，模拟每个个体的随机死亡
        for age, mortality in enumerate(self.mortality_rates):
            age_group = [person for person in old_population if person.age == age]
            num_deaths = int(len(age_group) * mortality)
            for _ in range(num_deaths):
                if age_group:
                    person_to_die = random.choice(age_group)
                    age_group.remove(person_to_die)
                    old_population.remove(person_to_die)
        
        # 模拟超出最大寿命而老死
        old_population = [person for person in old_population if person.age < self.max_age]
        
        # 更新死亡人口数量
        self.deaths = self.initial_population - len(old_population)
        
        # 获取不同年龄段的生育率，根据适龄婚育的女性人口数目，模拟新人口的出生
        num_children = 0
        for age, fertility in enumerate(self.fertility_rates):
            age_group = [person for person in old_population if person.age == age and person.gender == "female"]
            births = int(len(age_group) * fertility)
            num_children += births
            for _ in range(births):
                child = People()
                # 根据性别比，随机选择性别
                if np.random.random() < 0.525:
                    child.gender = "male"
                else:
                    child.gender = "female"
                child.age = 0
                new_population.append(child)
        
        # 更新出生人口数量
        self.births = num_children
        
        # 合并新出生人口和现有人口
        self.population = old_population + new_population
        
        # 记录当前状态
        self._record_state()
    
    def simulate(self):
        """模拟给定时间内的人口变化"""
        for _ in range(self.simulation_years):
            self.simulate_one_year()
            print(f"Year {self.initial_year + _ }: Total Population = {len(self.population)}")
        print(f"Simulation complete after {self.simulation_years} years. All states recorded.")
    
    def get_history(self):
        """获取历史数据作为DataFrame"""
        df = pd.DataFrame({
            'year': self.history['year_index'],
            'total_population': self.history['total_population'],
            'births': self.history['births'],
            'deaths': self.history['deaths'],
            'dependency_ratio': self.history['dependency_ratio'],
            'young_dependency_ratio': self.history['young_dependency_ratio'],
            'elderly_dependency_ratio': self.history['elderly_dependency_ratio'],
            'patriarchal_dependency_ratio': self.history['patriarchal_dependency_ratio'],
            'median_age': self.history['median_age'],
            'working_age_population': self.history['working_age_population'],
            'elderly_population': self.history['elderly_population'],
            'young_population': self.history['young_population']
        })
        return df
    
    def plot_population(self):
        """绘制人口变化图表"""
        history_df = self.get_history()
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. 总人口趋势
        axes[0, 0].plot(history_df['year'], history_df['total_population'], 'b-', linewidth=2)
        axes[0, 0].set_title('Population Trend')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Population')
        axes[0, 0].grid(True)
        
        # 2. 出生和死亡趋势
        axes[0, 1].plot(history_df['year'], history_df['births'], 'g-', label='births')
        axes[0, 1].plot(history_df['year'], history_df['deaths'], 'r-', label='deaths')
        axes[0, 1].set_title('Births and Deaths Trend')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Population')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 中位年龄
        axes[1, 0].plot(history_df['year'], history_df['median_age'], 'purple', linewidth=2)
        axes[1, 0].set_title('Median Age')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Age')
        axes[1, 0].grid(True)
        
        # 4. 抚养比
        axes[1, 1].plot(history_df['year'], history_df['elderly_dependency_ratio'], 'r-', label='Elderly Dependency Ratio')
        axes[1, 1].plot(history_df['year'], history_df['young_dependency_ratio'], 'g-', label='Young Dependency Ratio')
        axes[1, 1].plot(history_df['year'], history_df['dependency_ratio'], 'b--', label='Dependency Ratio')
        axes[1, 1].plot(history_df['year'], history_df['patriarchal_dependency_ratio'], 'm-.', label='Patriarchal Dependency Ratio')
        axes[1, 1].set_title('Dependency Ratio Trend')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 5. 人口结构变化
        working_age = history_df['working_age_population']
        elderly = history_df['elderly_population']
        young = history_df['young_population']
        
        axes[2, 0].stackplot(history_df['year'], 
                       [young, working_age, elderly],
                       labels=['young(0-14)', 'working age(15-64)', 'elderly(65+)'],
                       colors=['#3CB371', '#6495ED', '#DB7093'],
                       alpha=0.7)
        axes[2, 0].set_title('Age structure change over time')
        axes[2, 0].set_xlabel('Year')
        axes[2, 0].set_ylabel('Population')
        axes[2, 0].legend(loc='upper right')
        axes[2, 0].grid(True)
        
        # 6. 人口结构比例变化
        total_pop = history_df['total_population']
        working_age_pct = working_age / total_pop * 100
        elderly_pct = elderly / total_pop * 100
        young_pct = young / total_pop * 100
        
        axes[2, 1].stackplot(history_df['year'], 
                       [young_pct, working_age_pct, elderly_pct],
                       labels=['young(0-14)', 'working age(15-64)', 'elderly(65+)'],
                       colors=['#3CB371', '#6495ED', '#DB7093'],
                       alpha=0.7)
        axes[2, 1].set_title('Age structure proportion change over time')
        axes[2, 1].set_xlabel('Year')
        axes[2, 1].set_ylabel('Population Proportion (%)')
        axes[2, 1].set_ylim(0, 100)
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'RE_人口模拟_{self.scenario}_{self.birth_rate}.png')
        return fig, axes
    
    def plot_population_pyramid(self, year_idx=0, figsize=(10, 8)):
        """绘制特定年份的人口金字塔，year_idx接受int，表示自从初始年份开始的年份索引，例如20为20之后"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算实际年份
        actual_year = self.initial_year if year_idx == 0 else self.history['year_index'][year_idx]
        
        # 获取人口数据
        if year_idx == 0:
            population = self.population
        else:
            if year_idx >= len(self.history['age_distribution']):
                print(f"错误：请求的年份索引{year_idx}超出模拟范围")
                return None, None
            
            # 使用历史记录中的年龄分布和总人口
            age_dist = self.history['age_distribution'][year_idx]
            total_pop = self.history['total_population'][year_idx]
            
            # 创建虚拟人口以便于计算
            population = []
            for age in range(self.max_age):
                count = int(age_dist[age] * total_pop)
                for _ in range(count):
                    person = People()
                    person.age = age
                    if np.random.random() < 0.525:
                        person.gender = "male"
                    else:
                        person.gender = "female"
                    population.append(person)
        
        # 按年龄和性别统计人口
        age_groups = np.arange(0, self.max_age, 5)
        age_labels = [f"{i}-{i+4}" for i in age_groups[:-1]]
        age_labels.append(f"{age_groups[-1]}+")
        
        male_data = []
        female_data = []
        
        for i in range(len(age_groups)-1):
            start, end = age_groups[i], min(age_groups[i+1], self.max_age)
            males = len([p for p in population if p.age >= start and p.age < end and p.gender == "male"])
            females = len([p for p in population if p.age >= start and p.age < end and p.gender == "female"])
            male_data.append(males)
            female_data.append(females)
        
        # 最后一个年龄组
        males = len([p for p in population if p.age >= age_groups[-1] and p.gender == "male"])
        females = len([p for p in population if p.age >= age_groups[-1] and p.gender == "female"])
        male_data.append(males)
        female_data.append(females)
        
        # 创建横向条形图
        y_pos = np.arange(len(age_labels))
        ax.barh(y_pos, -np.array(male_data), color='steelblue', alpha=0.8, label='male')
        ax.barh(y_pos, female_data, color='lightcoral', alpha=0.8, label='female')
        
        # 设置图表标签和标题
        ax.set_title(f"Population pyramid at ({actual_year})")
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
        
        # 添加总人口信息
        total_pop = len(population)
        ax.text(0.02, 0.02, f"Population of {total_pop}",
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig(f'RE_人口金字塔_{self.scenario}_{actual_year}.png')
        return fig, ax
    
    def simulate(self):
        """模拟整个时间段内的人口变化并且记录数据"""
        self.initialize_population()
        for _ in range(self.simulation_years):
            self.simulate_one_year()
        
        return self.get_history()
    
    
if __name__ == "__main__":
        society = Society(birth_rate=0.78, initial_population=10000, simulation_years=50, scenario="reality")
        society.simulate()
        society.plot_population()
        society.plot_population_pyramid(year_idx=0)  # 初始年份
        society.plot_population_pyramid(year_idx=25)  # 中间年份
        society.plot_population_pyramid(year_idx=49)  # 最终年份