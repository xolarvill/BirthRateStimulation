import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class KoreaPopulationSimulator:
    def __init__(self, 
                 initial_population=51740000,  # 韩国2022年人口约5174万
                 initial_age_distribution=None,
                 max_age=100,
                 fertility_rates=None,
                 mortality_rates=None,
                 migration_rates=None,
                 simulation_years=70):
        """
        初始化韩国人口模拟器
        
        参数:
        - initial_population: 初始总人口数
        - initial_age_distribution: 初始年龄分布 (概率分布，总和为1)
        - max_age: 最大年龄
        - fertility_rates: 各年龄段的生育率 (按年龄索引的数组)
        - mortality_rates: 各年龄段的死亡率 (按年龄索引的数组)
        - migration_rates: 各年龄段的净迁移率 (按年龄索引的数组)
        - simulation_years: 模拟年数
        """
        self.max_age = max_age
        self.simulation_years = simulation_years
        self.year_offset = 2022  # 起始年份设为2022年
        
        # 设置韩国2022年的年龄分布
        if initial_age_distribution is None:
            # 使用更符合韩国人口结构的年龄分布
            # 韩国为老龄化社会，中年人口比例高，年轻人口比例低
            self.initial_age_distribution = self._get_korea_age_distribution()
        else:
            self.initial_age_distribution = initial_age_distribution
            
        # 设置韩国的生育率 - 总和生育率0.78
        if fertility_rates is None:
            # 韩国女性生育年龄主要集中在25-39岁，但总量很低
            self.fertility_rates = self._get_korea_fertility_rates()
        else:
            self.fertility_rates = fertility_rates
            
        # 设置韩国的死亡率
        if mortality_rates is None:
            # 韩国拥有较高的预期寿命，死亡率较低
            self.mortality_rates = self._get_korea_mortality_rates()
        else:
            self.mortality_rates = mortality_rates
            
        # 设置韩国的净迁移率
        if migration_rates is None:
            # 韩国净迁移率较低
            self.migration_rates = self._get_korea_migration_rates()
        else:
            self.migration_rates = migration_rates
            
        # 初始化人口
        self.population = np.round(initial_population * self.initial_age_distribution).astype(int)
        
        # 储存历史数据
        self.history = {
            'total_population': [],
            'age_distribution': [],
            'births': [],
            'deaths': [],
            'dependency_ratio': [],
            'median_age': [],
            'old_age_dependency': [],
            'youth_dependency': [],
            'working_age_population': [],
            'elderly_population': [],
            'children_population': [],
            'net_migration': []
        }
        
        # 记录初始状态
        self._record_state()
        
    def _get_korea_age_distribution(self):
        """返回韩国2022年的年龄分布估计"""
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
    
    def _get_korea_fertility_rates(self):
        """返回韩国2022年的生育率（总和生育率为0.78）"""
        fertility = np.zeros(self.max_age)
        
        # 韩国女性生育主要集中在25-39岁，但推迟生育趋势明显
        # 将0.78的总和生育率分配给不同年龄段
        fertility[20:25] = 0.01  # 20-24岁生育率很低
        fertility[25:30] = 0.08  # 25-29岁生育率开始上升
        fertility[30:35] = 0.20  # 30-34岁为生育高峰
        fertility[35:40] = 0.15  # 35-39岁仍有较高生育率
        fertility[40:45] = 0.05  # 40-44岁生育率降低
        fertility[45:50] = 0.01  # 45-49岁生育率很低
        
        # 确保总和生育率为0.78（需要除以2因为只有女性生育，假设性别比约为1:1）
        current_tfr = np.sum(fertility)
        fertility = fertility * (0.78 / current_tfr)  
        
        return fertility
    
    def _get_korea_mortality_rates(self):
        """返回韩国的死亡率（韩国拥有世界领先的预期寿命）"""
        mortality = np.zeros(self.max_age)
        
        # 婴儿死亡率（非常低）
        mortality[0] = 0.003
        
        # 儿童及青少年死亡率（极低）
        mortality[1:15] = 0.0005
        
        # 青年死亡率
        mortality[15:40] = 0.001
        
        # 中年死亡率（逐渐上升）
        mortality[40:65] = np.linspace(0.002, 0.01, 25)
        
        # 老年死亡率（加速上升）
        mortality[65:80] = np.linspace(0.015, 0.06, 15)
        mortality[80:90] = np.linspace(0.08, 0.2, 10)
        mortality[90:] = np.linspace(0.25, 0.5, self.max_age - 90)
        
        return mortality
    
    def _get_korea_migration_rates(self):
        """返回韩国的净迁移率（按年龄分布）"""
        migration = np.zeros(self.max_age)
        
        # 韩国净迁移率较低，但工作年龄人口有一定流入
        # 负值表示净流出，正值表示净流入
        migration[0:15] = -0.0005  # 儿童少量净流出
        migration[15:35] = 0.002   # 年轻工作人口净流入（留学生、专业人才等）
        migration[35:65] = 0.001   # 中年工作人口小量净流入
        migration[65:] = -0.001    # 老年人口小量净流出（退休移民等）
        
        return migration
        
    def set_fertility_scenario(self, scenario, custom_tfr=None):
        """
        设置生育率情景
        
        参数:
        - scenario: 情景名称 ('current', 'slight_increase', 'moderate_increase', 'recovery', 'custom')
        - custom_tfr: 如果选择'custom'情景，指定的总和生育率
        """
        base_rates = np.zeros(self.max_age)
        
        if scenario == 'current':
            # 维持韩国2022年的极低生育率0.78
            base_rates = self._get_korea_fertility_rates()
            print(f"设置情景: 维持当前总和生育率 0.78")
            
        elif scenario == 'slight_increase':
            # 轻微回升情景：总和生育率1.0
            base_rates[20:25] = 0.015
            base_rates[25:30] = 0.10
            base_rates[30:35] = 0.25
            base_rates[35:40] = 0.18
            base_rates[40:45] = 0.06
            base_rates[45:50] = 0.015
            # 归一化为总和生育率1.0
            current_tfr = np.sum(base_rates)
            base_rates = base_rates * (1.0 / current_tfr)
            print(f"设置情景: 轻微回升至总和生育率 1.0")
            
        elif scenario == 'moderate_increase':
            # 中度回升情景：总和生育率1.3
            base_rates[20:25] = 0.02
            base_rates[25:30] = 0.15
            base_rates[30:35] = 0.28
            base_rates[35:40] = 0.22
            base_rates[40:45] = 0.08
            base_rates[45:50] = 0.02
            # 归一化为总和生育率1.3
            current_tfr = np.sum(base_rates)
            base_rates = base_rates * (1.3 / current_tfr)
            print(f"设置情景: 中度回升至总和生育率 1.3")
            
        elif scenario == 'recovery':
            # 恢复至更替水平情景：总和生育率2.1
            base_rates[20:25] = 0.05
            base_rates[25:30] = 0.25
            base_rates[30:35] = 0.40
            base_rates[35:40] = 0.30
            base_rates[40:45] = 0.15
            base_rates[45:50] = 0.03
            # 归一化为总和生育率2.1
            current_tfr = np.sum(base_rates)
            base_rates = base_rates * (2.1 / current_tfr)
            print(f"设置情景: 恢复至更替水平总和生育率 2.1")
            
        elif scenario == 'custom':
            # 自定义生育率情景
            if custom_tfr is None:
                custom_tfr = 0.78  # 默认使用当前水平
                
            # 使用当前分布，但调整至自定义总和生育率
            base_rates[20:25] = 0.01
            base_rates[25:30] = 0.08
            base_rates[30:35] = 0.20
            base_rates[35:40] = 0.15
            base_rates[40:45] = 0.05
            base_rates[45:50] = 0.01
            
            current_tfr = np.sum(base_rates)
            base_rates = base_rates * (custom_tfr / current_tfr)
            print(f"设置情景: 自定义总和生育率 {custom_tfr}")
        
        self.fertility_rates = base_rates
        return np.sum(self.fertility_rates)  # 返回总和生育率
    
    def simulate_year(self):
        """模拟一年的人口变化"""
        # 计算新生儿
        female_ratio = 0.493  # 韩国出生性别比约为107:100，女性比例为0.493
        births = np.sum(self.population * self.fertility_rates * female_ratio)
        births = int(np.round(births))
        
        # 计算死亡人数
        deaths_by_age = np.random.binomial(self.population, self.mortality_rates)
        total_deaths = np.sum(deaths_by_age)
        
        # 计算净迁移人数（可正可负）
        migration_by_age = np.round(self.population * self.migration_rates).astype(int)
        net_migration = np.sum(migration_by_age)
        
        # 更新人口年龄结构
        new_population = np.zeros(self.max_age)
        new_population[0] = births  # 新生儿
        
        for age in range(1, self.max_age):
            # 当前年龄人口 = 上一年龄段存活人口 + 净迁移
            new_population[age] = max(0, self.population[age-1] - deaths_by_age[age-1] + migration_by_age[age-1])
        
        # 最高年龄段额外处理（防止人口无限增长超过最大年龄）
        new_population[self.max_age-1] += max(0, self.population[self.max_age-1] - deaths_by_age[self.max_age-1] + migration_by_age[self.max_age-1])
        
        # 更新当前人口
        self.population = np.maximum(0, new_population).astype(int)
        
        # 记录本年度状态
        self._record_state(net_migration)
        
        return births, total_deaths, net_migration
    
    def simulate(self):
        """模拟整个时间段"""
        for _ in range(self.simulation_years):
            self.simulate_year()
        
        return self.get_history()
    
    def _record_state(self, net_migration=0):
        """记录当前状态到历史数据"""
        total_pop = np.sum(self.population)
        
        if total_pop > 0:  # 防止除以零
            age_dist = self.population / total_pop
            children = np.sum(self.population[:15])
            working_age = np.sum(self.population[15:65])
            elderly = np.sum(self.population[65:])
            
            young_dependency = children / max(1, working_age)
            old_dependency = elderly / max(1, working_age)
            dependency_ratio = young_dependency + old_dependency
            
            # 计算中位年龄
            cumulative = np.cumsum(self.population)
            median_idx = np.searchsorted(cumulative, total_pop / 2)
            median_age = min(median_idx, self.max_age - 1)
        else:
            age_dist = np.zeros(self.max_age)
            dependency_ratio = 0
            median_age = 0
            children = 0
            working_age = 0
            elderly = 0
            young_dependency = 0
            old_dependency = 0
        
        # 更新历史数据
        self.history['total_population'].append(total_pop)
        self.history['age_distribution'].append(age_dist)
        self.history['births'].append(np.sum(self.population * self.fertility_rates * 0.493))
        self.history['deaths'].append(np.sum(self.population * self.mortality_rates))
        self.history['dependency_ratio'].append(dependency_ratio)
        self.history['median_age'].append(median_age)
        self.history['old_age_dependency'].append(old_dependency)
        self.history['youth_dependency'].append(young_dependency)
        self.history['working_age_population'].append(working_age)
        self.history['elderly_population'].append(elderly)
        self.history['children_population'].append(children)
        self.history['net_migration'].append(net_migration)
    
    def get_history(self):
        """获取历史数据作为DataFrame"""
        df = pd.DataFrame({
            'year': np.arange(self.year_offset, self.year_offset + len(self.history['total_population'])),
            'total_population': self.history['total_population'],
            'births': self.history['births'],
            'deaths': self.history['deaths'],
            'dependency_ratio': self.history['dependency_ratio'],
            'median_age': self.history['median_age'],
            'old_age_dependency': self.history['old_age_dependency'], 
            'youth_dependency': self.history['youth_dependency'],
            'working_age_population': self.history['working_age_population'],
            'elderly_population': self.history['elderly_population'],
            'children_population': self.history['children_population'],
            'net_migration': self.history['net_migration']
        })
        return df
    
    def plot_population_pyramid(self, year_idx=0, figsize=(10, 8), actual_year=None):
        """绘制特定年份的人口金字塔"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算实际年份
        if actual_year is None:
            actual_year = self.year_offset + year_idx
            
        # 获取数据
        if year_idx >= len(self.history['age_distribution']):
            print(f"错误：请求的年份索引{year_idx}超出模拟范围")
            return None, None
            
        population = self.population if year_idx == 0 else (self.history['age_distribution'][year_idx] * self.history['total_population'][year_idx]).astype(int)
        
        # 计算男女性别分布
        male_pop = population * 0.507  # 韩国男性比例约为50.7%
        female_pop = population * 0.493  # 韩国女性比例约为49.3%
        
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
        ax.barh(y_pos, -np.array(male_data), color='steelblue', alpha=0.8, label='male')
        ax.barh(y_pos, female_data, color='lightcoral', alpha=0.8, label='female')
        
        # 设置图表标签和标题
        ax.set_title(f"Korean population pyramid ({actual_year})")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(age_labels)
        ax.set_ylabel("age groups")
        ax.set_xlabel("population (k)")
        
        # 处理 x 轴标签以显示正值，并转换为千人单位
        ticks = ax.get_xticks()
        ax.set_xticklabels([str(abs(int(tick/1000))) for tick in ticks])
        
        # 添加图例
        ax.legend()
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加总人口信息
        total_pop = np.sum(population)
        ax.text(0.02, 0.02, f"total population: {total_pop/1000000:.2f}million",
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_demographic_trends(self, milestone_years=[30, 50, 70], figsize=(15, 12)):
        """
        绘制人口统计趋势图，并标记特定里程碑年份
        
        参数:
        - milestone_years: 需要在图表上特别标记的年数（从模拟开始算起）
        - figsize: 图表大小
        """
        history_df = self.get_history()
        
        # 将里程碑年份转换为实际年份
        milestone_actual_years = [self.year_offset + y for y in milestone_years]
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 设置共享颜色
        milestone_colors = ['red', 'green', 'purple']
        
        # 1. 总人口趋势
        axes[0, 0].plot(history_df['year'], history_df['total_population']/1000000, 'b-', linewidth=2)
        axes[0, 0].set_title('total population change')
        axes[0, 0].set_xlabel('year')
        axes[0, 0].set_ylabel('population (million)')
        
        # 标记里程碑年份
        for i, year in enumerate(milestone_actual_years):
            if year in history_df['year'].values:
                idx = history_df[history_df['year'] == year].index[0]
                pop = history_df.iloc[idx]['total_population']/1000000
                axes[0, 0].axvline(x=year, color=milestone_colors[i], linestyle='--', alpha=0.7)
                axes[0, 0].text(year+0.5, pop, f"{year}: {pop:.2f} million", color=milestone_colors[i])
        
        axes[0, 0].grid(True)
        
        # 2. 出生和死亡趋势
        axes[0, 1].plot(history_df['year'], history_df['births']/1000, 'g-', label='births (k)')
        axes[0, 1].plot(history_df['year'], history_df['deaths']/1000, 'r-', label='deaths (k)')
        axes[0, 1].set_title('trend of births and deaths')
        axes[0, 1].set_xlabel('year')
        axes[0, 1].set_ylabel('number (k)')
        
        for i, year in enumerate(milestone_actual_years):
            if year in history_df['year'].values:
                idx = history_df[history_df['year'] == year].index[0]
                axes[0, 1].axvline(x=year, color=milestone_colors[i], linestyle='--', alpha=0.7)
        
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 中位年龄
        axes[1, 0].plot(history_df['year'], history_df['median_age'], 'purple', linewidth=2)
        axes[1, 0].set_title('population median age')
        axes[1, 0].set_xlabel('year')
        axes[1, 0].set_ylabel('age')
        
        for i, year in enumerate(milestone_actual_years):
            if year in history_df['year'].values:
                idx = history_df[history_df['year'] == year].index[0]
                median_age = history_df.iloc[idx]['median_age']
                axes[1, 0].axvline(x=year, color=milestone_colors[i], linestyle='--', alpha=0.7)
                axes[1, 0].text(year+0.5, median_age, f"{year}: {median_age:.1f}岁", color=milestone_colors[i])
        
        axes[1, 0].grid(True)
        
        # 4. 抚养比
        axes[1, 1].plot(history_df['year'], history_df['old_age_dependency'], 'r-', label='ratio of rearing elderly')
        axes[1, 1].plot(history_df['year'], history_df['youth_dependency'], 'g-', label='ratio of rearing young')
        axes[1, 1].plot(history_df['year'], history_df['dependency_ratio'], 'b--', label='ratio of rearing')
        axes[1, 1].set_title('change in ratio of rearing')
        axes[1, 1].set_xlabel('year')
        axes[1, 1].set_ylabel('ratio')
        
        for i, year in enumerate(milestone_actual_years):
            if year in history_df['year'].values:
                idx = history_df[history_df['year'] == year].index[0]
                axes[1, 1].axvline(x=year, color=milestone_colors[i], linestyle='--', alpha=0.7)
        
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 5. 人口结构变化
        working_age = history_df['working_age_population']/1000000
        elderly = history_df['elderly_population']/1000000
        children = history_df['children_population']/1000000
        
        axes[2, 0].stackplot(history_df['year'], 
                           [children, working_age, elderly],
                           labels=['minor(0-14)', 'mid(15-64)', 'elderly(65+)'],
                           colors=['#3CB371', '#6495ED', '#DB7093'],
                           alpha=0.7)
        axes[2, 0].set_title('change of population structure')
        axes[2, 0].set_xlabel('year')
        axes[2, 0].set_ylabel('population (million)')
        
        for i, year in enumerate(milestone_actual_years):
            if year in history_df['year'].values:
                axes[2, 0].axvline(x=year, color=milestone_colors[i], linestyle='--', alpha=0.7)
        
        axes[2, 0].legend(loc='upper right')
        axes[2, 0].grid(True)
        
        # 6. 人口结构比例变化
        total_pop = history_df['total_population']
        working_age_pct = history_df['working_age_population'] / total_pop * 100
        elderly_pct = history_df['elderly_population'] / total_pop * 100
        children_pct = history_df['children_population'] / total_pop * 100
        
        axes[2, 1].stackplot(history_df['year'], 
                           [children_pct, working_age_pct, elderly_pct],
                           labels=['minor(0-14)', 'mid(15-64)', 'elderly(65+)'],
                           colors=['#3CB371', '#6495ED', '#DB7093'],
                           alpha=0.7)
        axes[2, 1].set_title('change of population structure')
        axes[2, 1].set_xlabel('year')
        axes[2, 1].set_ylabel('population proportion（%）')
        axes[2, 1].set_ylim(0, 100)
        
        for i, year in enumerate(milestone_actual_years):
            if year in history_df['year'].values:
                idx = history_df[history_df['year'] == year].index[0]
                elderly_val = elderly_pct.iloc[idx]
                axes[2, 1].axvline(x=year, color=milestone_colors[i], linestyle='--', alpha=0.7)
                axes[2, 1].text(year+0.5, 80, f"{year}: 老年{elderly_val:.1f}%", color=milestone_colors[i])
        
        # 避免图例重复
        # axes[2, 1].legend(loc='upper right')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        return fig, axes
    
    def milestone_analysis(self, milestones=[30, 50, 70]):
        """
        对里程碑年份进行详细分析
        返回关键指标的数据框
        """
        history_df = self.get_history()
        milestone_years = [self.year_offset + m for m in milestones]
        
        # 创建结果数据框
        results = []
        
        # 基准年份(2022)数据
        base_data = history_df.iloc[0]
        results.append({
            '年份': self.year_offset,
            '总人口(百万)': base_data['total_population'] / 1000000,
            '出生人数(千)': base_data['births'] / 1000,
            '死亡人数(千)': base_data['deaths'] / 1000,
            '人口出生率(‰)': base_data['births'] / base_data['total_population'] * 1000,
            '人口死亡率(‰)': base_data['deaths'] / base_data['total_population'] * 1000,
            '人口自然增长率(‰)': (base_data['births'] - base_data['deaths']) / base_data['total_population'] * 1000,
            '中位年龄': base_data['median_age'],
            '老年抚养比': base_data['old_age_dependency'],
            '少儿抚养比': base_data['youth_dependency'],
            '总抚养比': base_data['dependency_ratio'],
            '工作年龄人口比例(%)': base_data['working_age_population'] / base_data['total_population'] * 100,
            '老年人口比例(%)': base_data['elderly_population'] / base_data['total_population'] * 100,
            '儿童人口比例(%)': base_data['children_population'] / base_data['total_population'] * 100,
            '净迁移人数': base_data['net_migration']
        })
        
        # 里程碑年份数据
        for year in milestone_years:
            if year in history_df['year'].values:
                year_data = history_df[history_df['year'] == year].iloc[0]
                results.append({
                    '年份': year,
                    '总人口(百万)': year_data['total_population'] / 1000000,
                    '出生人数(千)': year_data['births'] / 1000,
                    '死亡人数(千)': year_data['deaths'] / 1000,
                    '人口出生率(‰)': year_data['births'] / year_data['total_population'] * 1000,
                    '人口死亡率(‰)': year_data['deaths'] / year_data['total_population'] * 1000,
                    '人口自然增长率(‰)': (year_data['births'] - year_data['deaths']) / year_data['total_population'] * 1000,
                    '中位年龄': year_data['median_age'],
                    '老年抚养比': year_data['old_age_dependency'],
                    '少儿抚养比': year_data['youth_dependency'],
                    '总抚养比': year_data['dependency_ratio'],
                    '工作年龄人口比例(%)': year_data['working_age_population'] / year_data['total_population'] * 100,
                    '老年人口比例(%)': year_data['elderly_population'] / year_data['total_population'] * 100,
                    '儿童人口比例(%)': year_data['children_population'] / year_data['total_population'] * 100,
                    '净迁移人数': year_data['net_migration']
                })
        
        return pd.DataFrame(results)
    
    def create_animation(self, interval_years=5, figsize=(10, 8)):
        """
        创建人口金字塔动画
        
        参数:
        - interval_years: 动画中显示的年份间隔
        - figsize: 图表大小
        
        返回:
        - HTML动画对象
        """
        # 选择要显示的年份
        years_to_show = list(range(0, len(self.history['age_distribution']), interval_years))
        if (len(self.history['age_distribution'])-1) not in years_to_show:
            years_to_show.append(len(self.history['age_distribution'])-1)  # 确保包含最后一年
            
        fig, ax = plt.subplots(figsize=figsize)
        
        def update(frame):
            ax.clear()
            year_idx = years_to_show[frame]
            actual_year = self.year_offset + year_idx
            
            # 获取数据
            population = (self.history['age_distribution'][year_idx] * self.history['total_population'][year_idx]).astype(int)
            
            # 计算男女性别分布
            male_pop = population * 0.507  # 韩国男性比例约为50.7%
            female_pop = population * 0.493  # 韩国女性比例约为49.3%
            
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
            ax.barh(y_pos, -np.array(male_data), color='steelblue', alpha=0.8, label='male')
            ax.barh(y_pos, female_data, color='lightcoral', alpha=0.8, label='female')
            
            # 设置图表标签和标题
            ax.set_title(f"Korean population pyramid ({actual_year})")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(age_labels)
            ax.set_ylabel("age groups")
            ax.set_xlabel("population (k)")
            
            # 处理 x 轴标签以显示正值
            ticks = ax.get_xticks()
            ax.set_xticklabels([str(abs(int(tick/1000))) for tick in ticks])
            
            # 添加图例
            ax.legend()
            
            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 添加总人口信息
            total_pop = np.sum(population)
            ax.text(0.02, 0.02, f"population: {total_pop/1000000:.2f} million",
                    transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
            
            return ax,
        
        # 创建动画
        ani = FuncAnimation(fig, update, frames=len(years_to_show), interval=1000, blit=False)
        plt.close()  # 防止显示静态图
        
        return HTML(ani.to_jshtml())
    
    def set_migration_scenario(self, scenario, custom_factor=None):
        """
        设置迁移情景
        
        参数:
        - scenario: 情景名称 ('current', 'increased', 'massive_immigration', 'custom')
        - custom_factor: 如果选择'custom'情景，指定的迁移因子
        """
        if scenario == 'current':
            # 维持当前迁移水平
            self.migration_rates = self._get_korea_migration_rates()
            print(f"设置情景: 维持当前迁移率")
            
        elif scenario == 'increased':
            # 增加迁移 - 基准的3倍
            base_rates = self._get_korea_migration_rates()
            self.migration_rates = base_rates * 3
            print(f"设置情景: 迁移率提高至基准的3倍")
            
        elif scenario == 'massive_immigration':
            # 大规模移民 - 基准的10倍，且主要针对工作年龄人口
            base_rates = self._get_korea_migration_rates()
            self.migration_rates = base_rates * 10
            
            # 增强工作年龄人口移民
            self.migration_rates[20:40] *= 2  # 进一步增加青年工作人口移民
            print(f"设置情景: 大规模移民政策 (基准的10倍，工作年龄人口更高)")
            
        elif scenario == 'custom':
            # 自定义迁移率
            if custom_factor is None:
                custom_factor = 1.0  # 默认为基准水平
                
            base_rates = self._get_korea_migration_rates()
            self.migration_rates = base_rates * custom_factor
            print(f"设置情景: 自定义迁移因子 {custom_factor}")
            
        return self.migration_rates
    
    def compare_scenarios(self, scenario_list, figsize=(15, 10)):
        """
        比较不同的人口情景
        
        参数:
        - scenario_list: 情景列表，每个元素为dict，需包含keys: 'name', 'fertility', 'migration', 'custom_tfr', 'custom_migration'
        - figsize: 图表大小
        
        返回:
        - fig, axes: 图表对象
        """
        # 存储原始状态
        original_fertility = self.fertility_rates.copy()
        original_migration = self.migration_rates.copy()
        original_population = self.population.copy()
        original_history = self.history.copy()
        
        # 准备结果存储
        scenario_results = []
        
        # 运行每个情景
        for scenario in scenario_list:
            # 重置模拟器状态
            self.population = original_population.copy()
            self.history = {key: original_history[key].copy() for key in original_history}
            
            # 设置生育率
            if 'fertility' in scenario:
                custom_tfr = scenario.get('custom_tfr', None)
                self.set_fertility_scenario(scenario['fertility'], custom_tfr)
                
            # 设置迁移率
            if 'migration' in scenario:
                custom_factor = scenario.get('custom_migration', None)
                self.set_migration_scenario(scenario['migration'], custom_factor)
                
            # 运行模拟
            self.simulate()
            
            # 存储结果
            scenario_results.append({
                'name': scenario['name'],
                'history': self.get_history()
            })
            
        # 恢复原始状态
        self.fertility_rates = original_fertility
        self.migration_rates = original_migration
        self.population = original_population
        self.history = original_history
        
        # 创建比较图表
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 定义颜色列表
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        # 1. 总人口比较
        for i, scenario in enumerate(scenario_results):
            color = colors[i % len(colors)]
            df = scenario['history']
            axes[0, 0].plot(df['year'], df['total_population']/1000000, 
                           color=color, label=scenario['name'], linewidth=2)
            
        axes[0, 0].set_title('comparison of total population')
        axes[0, 0].set_xlabel('year')
        axes[0, 0].set_ylabel('population (million)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 老年人口比例比较
        for i, scenario in enumerate(scenario_results):
            color = colors[i % len(colors)]
            df = scenario['history']
            elderly_pct = df['elderly_population'] / df['total_population'] * 100
            axes[0, 1].plot(df['year'], elderly_pct, 
                           color=color, label=scenario['name'], linewidth=2)
            
        axes[0, 1].set_title('comparison of elderly proportion')
        axes[0, 1].set_xlabel('year')
        axes[0, 1].set_ylabel('elderly proportion（%）')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 总抚养比比较
        for i, scenario in enumerate(scenario_results):
            color = colors[i % len(colors)]
            df = scenario['history']
            axes[1, 0].plot(df['year'], df['dependency_ratio'], 
                           color=color, label=scenario['name'], linewidth=2)
            
        axes[1, 0].set_title('comparison of ratio of rearing')
        axes[1, 0].set_xlabel('year')
        axes[1, 0].set_ylabel('ratio of rearing')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. 中位年龄比较
        for i, scenario in enumerate(scenario_results):
            color = colors[i % len(colors)]
            df = scenario['history']
            axes[1, 1].plot(df['year'], df['median_age'], 
                           color=color, label=scenario['name'], linewidth=2)
            
        axes[1, 1].set_title('comparison of median age')
        axes[1, 1].set_xlabel('year')
        axes[1, 1].set_ylabel('age')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig, axes, scenario_results
    
    def generate_report(self, scenario_name="默认情景"):
        """生成人口预测报告"""
        history_df = self.get_history()
        
        # 计算关键年份的数据
        start_year = history_df['year'].min()
        end_year = history_df['year'].max()
        start_data = history_df.iloc[0]
        end_data = history_df.iloc[-1]
        
        # 计算人口变化率
        pop_change = (end_data['total_population'] - start_data['total_population']) / start_data['total_population'] * 100
        
        # 计算关键比例的变化
        old_ratio_start = start_data['elderly_population'] / start_data['total_population'] * 100
        old_ratio_end = end_data['elderly_population'] / end_data['total_population'] * 100
        
        work_ratio_start = start_data['working_age_population'] / start_data['total_population'] * 100
        work_ratio_end = end_data['working_age_population'] / end_data['total_population'] * 100
        
        # 计算预期生育率
        tfr = np.sum(self.fertility_rates)
        
        # 生成报告文本
        report = f"""
        # 韩国人口预测报告: {scenario_name}
        
        ## 模拟概述
        - 模拟期间: {start_year}年 - {end_year}年 (共{end_year-start_year}年)
        - 总和生育率设定: {tfr:.2f}
        
        ## 关键发现
        
        ### 人口规模
        - {start_year}年人口: {start_data['total_population']/1000000:.2f}百万
        - {end_year}年人口: {end_data['total_population']/1000000:.2f}百万
        - 人口变化: {pop_change:.1f}% ({pop_change>0 and '增长' or '减少'})
        
        ### 人口结构
        - 人口中位年龄: {start_data['median_age']:.1f}岁 -> {end_data['median_age']:.1f}岁
        - 老年人口(65岁以上)比例: {old_ratio_start:.1f}% -> {old_ratio_end:.1f}%
        - 工作年龄人口(15-64岁)比例: {work_ratio_start:.1f}% -> {work_ratio_end:.1f}%
        - 总抚养比: {start_data['dependency_ratio']:.2f} -> {end_data['dependency_ratio']:.2f}
        
        ### 人口动态
        - 模拟期末出生率: {end_data['births']/end_data['total_population']*1000:.2f}‰
        - 模拟期末死亡率: {end_data['deaths']/end_data['total_population']*1000:.2f}‰
        - 模拟期末自然增长率: {(end_data['births']-end_data['deaths'])/end_data['total_population']*1000:.2f}‰
        
        ## 社会影响分析
        
        ### 劳动力市场
        - 工作年龄人口减少对经济的影响: {'中等' if pop_change > -20 else '严重'}
        - 预计劳动力短缺程度: {'中等' if work_ratio_end > 50 else '严重'}
        
        ### 养老系统
        - 预计养老压力: {'中等' if end_data['old_age_dependency'] < 0.5 else '严重'}
        - 老年人口比例超过总人口30%的年份: {self.year_when_elderly_exceeds(30)}
        
        ### 综合评估
        - 人口规模走势: {'稳定' if abs(pop_change) < 10 else '下降' if pop_change < 0 else '增长'}
        - 人口结构走势: {'平衡' if end_data['old_age_dependency'] < 0.4 else '老龄化' if end_data['old_age_dependency'] < 0.6 else '超老龄化'}
        - 人口红利状况: {'持续' if work_ratio_end > 65 else '消失'}
        """
        
        return report
    
    def year_when_elderly_exceeds(self, threshold_percent):
        """
        确定老年人口比例超过特定阈值的年份
        
        参数:
        - threshold_percent: 比例阈值（如30表示30%）
        
        返回:
        - 超过阈值的年份，如果未超过则返回"超出模拟范围"
        """
        history_df = self.get_history()
        elderly_pct = history_df['elderly_population'] / history_df['total_population'] * 100
        
        # 找到第一个超过阈值的年份
        exceed_years = history_df[elderly_pct > threshold_percent]
        
        if len(exceed_years) > 0:
            return exceed_years['year'].iloc[0]
        else:
            return "超出模拟范围"
    
    def visualize_age_distribution_change(self, years=[0, 30, 50, 70], figsize=(12, 8)):
        """
        可视化不同年份的年龄分布变化
        
        参数:
        - years: 要比较的年份索引列表
        - figsize: 图表大小
        
        返回:
        - 图表对象
        """
        history_df = self.get_history()
        fig, ax = plt.subplots(figsize=figsize)
        
        # 过滤掉超出范围的年份
        valid_years = [y for y in years if y < len(history_df)]
        
        # 生成年龄标签
        age_labels = [f"{i}" for i in range(0, self.max_age, 5)]
        age_indices = range(0, self.max_age, 5)
        
        # 设置不同线型和颜色
        linestyles = ['-', '--', '-.', ':']
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, year_idx in enumerate(valid_years):
            # 获取该年的年龄分布
            year = history_df['year'].iloc[year_idx]
            age_dist = self.history['age_distribution'][year_idx] * 100
            
            # 每5岁绘制一个点以减少密度
            ax.plot(age_indices, age_dist[age_indices], 
                   label=f"year {year}", 
                   linestyle=linestyles[i % len(linestyles)],
                   color=colors[i % len(colors)],
                   linewidth=2,
                   marker='o',
                   markersize=5)
        
        ax.set_title('changes in korean age distribution韩国人口年龄分布变化')
        ax.set_xlabel('age')
        ax.set_ylabel('population proportion (%)')
        ax.grid(True)
        ax.legend()
        
        # 添加垂直线标记重要年龄段
        ax.axvline(x=15, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=65, color='gray', linestyle='--', alpha=0.5)
        ax.text(15, ax.get_ylim()[1]*0.95, '15岁', color='gray')
        ax.text(65, ax.get_ylim()[1]*0.95, '65岁', color='gray')
        
        plt.tight_layout()
        return fig, ax

# 使用示例
if __name__ == "__main__":
    society = 'south_korean'
    # 创建模拟器实例
    simulator = KoreaPopulationSimulator()
    
    # 设置生育率情景
    simulator.set_fertility_scenario('current')  # 维持当前极低生育率
    
    # 运行模拟
    history = simulator.simulate()
    
    # 显示人口金字塔
    plt.figure(figsize=(12, 8))
    simulator.plot_population_pyramid(year_idx=0)
    plt.title("korean_2022_pyramids")
    plt.savefig(f'{society}韩国2022年人口金字塔')
    
    plt.figure(figsize=(12, 8))
    simulator.plot_population_pyramid(year_idx=50)
    plt.title("korean_2072_pyramids")
    plt.savefig(f'{society}韩国2072年人口金字塔')
    
    # 显示人口趋势图
    fig, _ = simulator.plot_demographic_trends()
    plt.savefig(f'{society}显示人口趋势图')
    
    # 获取里程碑年份分析
    milestone_df = simulator.milestone_analysis()
    print(milestone_df)
    
    # 比较不同情景
    scenarios = [
        {'name': '当前极低生育率', 'fertility': 'current'},
        {'name': '生育率轻微回升', 'fertility': 'slight_increase'},
        {'name': '生育率恢复更替水平', 'fertility': 'recovery'}
    ]
    
    fig, _, _ = simulator.compare_scenarios(scenarios)
    plt.savefig(f'{society}比较不同情景')
    
    # 生成报告
    report = simulator.generate_report("基准情景：维持极低生育率")
    with open("population_report.md", "w", encoding="utf-8") as file:
        file.write(report)