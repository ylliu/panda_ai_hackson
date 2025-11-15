"""
期货回测框架

特点：
1. 完整的保证金、杠杆、强平机制
2. 详细的交易成本计算（手续费、滑点、平今仓）
3. 合约参数配置（乘数、最小跳动单位等）
4. 完整的风险控制（止损、止盈、强平）
5. 全面的绩效统计指标
"""
import panda_data
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# 可视化相关导入
try:
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.dates as mdates
    from matplotlib import font_manager
    
    # 设置中文字体 - 强力版本，确保中文正确显示
    import os
    import platform
    
    system = platform.system()
    chinese_font_name = None
    chinese_font_path = None
    chinese_font_prop = None
    
    # 根据系统查找字体文件路径（最可靠的方法）
    font_paths = []
    if system == 'Windows':
        windir = os.environ.get('WINDIR', 'C:\\Windows')
        font_paths = [
            (os.path.join(windir, 'Fonts', 'msyh.ttc'), 'Microsoft YaHei'),  # 微软雅黑
            (os.path.join(windir, 'Fonts', 'msyhbd.ttc'), 'Microsoft YaHei Bold'),  # 微软雅黑粗体
            (os.path.join(windir, 'Fonts', 'simhei.ttf'), 'SimHei'),  # 黑体
            (os.path.join(windir, 'Fonts', 'simsun.ttc'), 'SimSun'),  # 宋体
            (os.path.join(windir, 'Fonts', 'simkai.ttf'), 'KaiTi'),  # 楷体
        ]
    elif system == 'Darwin':  # macOS
        font_paths = [
            ('/System/Library/Fonts/PingFang.ttc', 'PingFang SC'),
            ('/System/Library/Fonts/STHeiti Light.ttc', 'STHeiti'),
            ('/Library/Fonts/Arial Unicode.ttf', 'Arial Unicode MS'),
        ]
    elif system == 'Linux':
        font_paths = [
            ('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 'WenQuanYi Micro Hei'),
            ('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 'WenQuanYi Zen Hei'),
            ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 'Noto Sans CJK SC'),
        ]
    
    # 优先通过字体文件路径查找
    for font_path, font_name in font_paths:
        if os.path.exists(font_path):
            try:
                # 直接使用字体文件创建FontProperties
                chinese_font_prop = font_manager.FontProperties(fname=font_path)
                chinese_font_name = font_name
                chinese_font_path = font_path
                print(f"✓ 已找到中文字体文件: {font_name} ({font_path})")
                break
            except Exception as e:
                continue
    
    # 如果通过文件路径没找到，尝试从已注册字体中查找
    if chinese_font_prop is None:
        chinese_font_names = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
                             'STSong', 'STKaiti', 'STHeiti', 'YouYuan', 'PingFang SC',
                             'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
        
        available_fonts = {f.name: f for f in font_manager.fontManager.ttflist}
        
        for font_name in chinese_font_names:
            if font_name in available_fonts:
                try:
                    font_file = available_fonts[font_name].fname
                    if os.path.exists(font_file):
                        chinese_font_prop = font_manager.FontProperties(fname=font_file)
                        chinese_font_name = font_name
                        chinese_font_path = font_file
                        print(f"✓ 已找到中文字体: {font_name}")
                        break
                except:
                    continue
    
    # 设置字体列表
    if chinese_font_name:
        font_list = [chinese_font_name, 'Arial Unicode MS', 'DejaVu Sans']
        print(f"✓ 已设置中文字体: {chinese_font_name}")
    else:
        font_list = ['Arial Unicode MS', 'DejaVu Sans']
        print("⚠ 警告: 未找到中文字体，可能无法正确显示中文")
        # 尝试创建默认字体属性
        chinese_font_prop = font_manager.FontProperties()
    
    # 清除matplotlib字体缓存（如果存在）
    try:
        cache_dir = matplotlib.get_cachedir()
        font_cache_pattern = os.path.join(cache_dir, 'fontlist-*.json')
        import glob
        cache_files = glob.glob(font_cache_pattern)
        if cache_files:
            for cache_file in cache_files:
                try:
                    os.remove(cache_file)
                    print(f"✓ 已清除字体缓存: {os.path.basename(cache_file)}")
                except:
                    pass
    except:
        pass
    
    # 设置全局字体参数
    plt.rcParams['font.sans-serif'] = font_list
    matplotlib.rcParams['font.sans-serif'] = font_list
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    # 保存字体属性供绘图函数使用
    CHINESE_FONT_PROP = chinese_font_prop if chinese_font_prop else font_manager.FontProperties()
    
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    CHINESE_FONT_PROP = None  # 如果matplotlib未安装，设置为None
    print("警告: matplotlib未安装，无法生成图表")


def get_chinese_font_prop():
    """安全获取中文字体属性"""
    if HAS_MATPLOTLIB and CHINESE_FONT_PROP is not None:
        return CHINESE_FONT_PROP
    elif HAS_MATPLOTLIB:
        # 如果字体属性未设置，尝试创建默认的
        try:
            from matplotlib import font_manager
            return font_manager.FontProperties()
        except:
            return None
    return None


def generate_ma_signals(
        df_data: pd.DataFrame,
        ma_short: int = 5,
        ma_long: int = 20,
        min_periods: int = 1
) -> pd.DataFrame:
    """
    基于 close 价格生成均线交叉信号

    参数:
    - df_data: pd.DataFrame, 必须包含 'close' 列，index 为 datetime
    - ma_short: 短均线周期
    - ma_long: 长均线周期
    - min_periods: 最小计算周期（默认取 ma_short）

    返回:
    - df_data: pd.DataFrame, 在原有数据基础上添加 'signal' 字段
        signal 值 ∈ {-1, 0, 1}
        1  = 金叉 → 开多
       -1  = 死叉 → 开空
        0  = 止盈止损 20分钟内无金叉死叉平仓
    """
    if min_periods is None:
        min_periods = ma_short

    # 计算均线
    df_data['ma_s'] = df_data['close'].rolling(window=ma_short, min_periods=min_periods).mean()
    df_data['ma_l'] = df_data['close'].rolling(window=ma_long, min_periods=min_periods).mean()

    # 初始化信号序列
    signal = pd.Series(0, index=df_data.index, dtype=int)

    # 检测金叉和死叉
    # 金叉：短均线上穿长均线（当前 ma_s > ma_l 且前一个 ma_s <= ma_l）
    golden_cross = (df_data['ma_s'] > df_data['ma_l']) & (df_data['ma_s'].shift(1) <= df_data['ma_l'].shift(1))

    # 死叉：短均线下穿长均线（当前 ma_s < ma_l 且前一个 ma_s >= ma_l）
    death_cross = (df_data['ma_s'] < df_data['ma_l']) & (df_data['ma_s'].shift(1) >= df_data['ma_l'].shift(1))

    # 标记交叉点
    signal[golden_cross] = 1  # 金叉 → 开多
    signal[death_cross] = -1  # 死叉 → 开空

    # 处理20分钟内无金叉死叉平仓的逻辑
    # 交叉点之后，信号保持，直到20分钟后没有新的交叉，则平仓（信号为0）
    window_minutes = 20

    # 找到所有交叉点的索引位置和对应的信号值
    cross_indices = signal[signal != 0].index

    if len(cross_indices) > 0:
        if isinstance(df_data.index, pd.DatetimeIndex):
            # 时间索引：使用时间差
            # 对于每个时间点，找到最近的交叉点并保持其信号
            for idx in df_data.index:
                if idx in cross_indices:
                    continue  # 交叉点本身保留信号

                # 找到最近的交叉点（向前查找）
                past_crosses = cross_indices[cross_indices < idx]
                if len(past_crosses) > 0:
                    # 找到最近的过去交叉点
                    nearest_cross = past_crosses.max()
                    time_diff = idx - nearest_cross
                    # 如果最近交叉点在20分钟内，保持信号
                    if time_diff <= pd.Timedelta(minutes=window_minutes):
                        signal[idx] = signal[nearest_cross]
                    # 否则平仓（信号已经是0，不需要修改）
        else:
            # 位置索引：使用位置差
            cross_positions = np.array([df_data.index.get_loc(ci) for ci in cross_indices])
            for i, idx in enumerate(df_data.index):
                if idx in cross_indices:
                    continue  # 交叉点本身保留信号

                # 找到最近的过去交叉点位置
                past_crosses = cross_positions[cross_positions < i]
                if len(past_crosses) > 0:
                    nearest_cross_pos = past_crosses.max()
                    pos_diff = i - nearest_cross_pos
                    # 如果最近交叉点在20个数据点内，保持信号
                    if pos_diff <= window_minutes:
                        nearest_cross_idx = cross_indices[cross_positions == nearest_cross_pos][0]
                        signal[idx] = signal[nearest_cross_idx]
                    # 否则平仓（信号已经是0，不需要修改）

    # 处理重复信号：1之后的所有1设为NaN，直到出现-1或0；-1之后的所有-1设为NaN，直到出现1或0；0之后的所有0设为NaN，直到出现1或-1
    # 即使是交叉点，如果与上一个非NaN信号相同，也设为NaN
    # last_signal = None
    # for idx in df_data.index:
    #     current_signal = signal[idx]
    #
    #     # 跳过已经是NaN的信号
    #     if pd.isna(current_signal):
    #         continue
    #
    #     if last_signal is not None and current_signal == last_signal:
    #         # 如果当前信号与上一个非NaN信号相同，设为NaN（包括交叉点）
    #         signal[idx] = np.nan
    #     else:
    #         # 如果信号改变，保留并更新last_signal
    #         last_signal = current_signal

    # 将 signal 添加到 df_data 中
    df_data['signal'] = signal

    return df_data


class OrderSide(Enum):
    """订单方向"""
    LONG = 1
    SHORT = -1
    CLOSE = 0


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"  # 资金不足等


@dataclass
class ContractConfig:
    """合约配置"""
    symbol: str
    multiplier: float = 10.0  # 合约乘数（每点价值）
    tick_size: float = 1.0  # 最小跳动单位
    margin_ratio: float = 0.12  # 保证金比例（如12%）
    commission_rate: float = 0.00023  # 手续费率（按金额）
    commission_per_contract: float = 0.0  # 固定手续费（按手）
    close_today_commission_rate: float = 0.0  # 平今仓手续费率（0表示与普通平仓相同）
    close_today_commission_per_contract: float = 0.0  # 平今仓固定手续费
    max_position: int = 5000  # 单品种持仓限额
    trade_unit: int = 1  # 最小交易手数
    
    @property
    def price_tick_value(self) -> float:
        """每跳价值 = tick_size × multiplier"""
        return self.tick_size * self.multiplier


@dataclass
class FuturePosition:
    """期货持仓"""
    side: OrderSide  # LONG or SHORT
    entry_price: float
    entry_timestamp: pd.Timestamp
    quantity: int  # 手数（整数）
    contract: ContractConfig
    
    # 止盈止损参数
    stop_loss_points: Optional[float] = None  # 固定止损点数
    stop_loss_percent: Optional[float] = None  # 百分比止损
    take_profit_points: Optional[float] = None  # 固定止盈点数
    take_profit_percent: Optional[float] = None  # 百分比止盈
    trailing_stop_pct: Optional[float] = None  # 移动止损百分比
    
    # 动态跟踪
    highest_price: float = field(default=None)
    lowest_price: float = field(default=None)
    is_today_position: bool = True  # 是否今日开仓（用于平今仓判断）
    
    def __post_init__(self):
        if self.highest_price is None:
            self.highest_price = self.entry_price
        if self.lowest_price is None:
            self.lowest_price = self.entry_price
    
    def get_margin_required(self, current_price: float) -> float:
        """计算所需保证金"""
        contract_value = current_price * self.contract.multiplier * self.quantity
        return contract_value * self.contract.margin_ratio
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """计算浮动盈亏（绝对金额）"""
        if self.side == OrderSide.LONG:
            pnl = (current_price - self.entry_price) * self.contract.multiplier * self.quantity
        else:  # SHORT
            pnl = (self.entry_price - current_price) * self.contract.multiplier * self.quantity
        return pnl
    
    def update_price(self, current_price: float):
        """更新价格并跟踪最高/最低价"""
        if self.side == OrderSide.LONG:
            self.highest_price = max(self.highest_price, current_price)
        else:  # SHORT
            self.lowest_price = min(self.lowest_price, current_price)
    
    def check_stop_loss(self, current_price: float) -> bool:
        """检查是否触发止损"""
        if self.stop_loss_points is not None:
            if self.side == OrderSide.LONG:
                return current_price <= self.entry_price - self.stop_loss_points
            else:
                return current_price >= self.entry_price + self.stop_loss_points
        
        if self.stop_loss_percent is not None:
            if self.side == OrderSide.LONG:
                return current_price <= self.entry_price * (1 - self.stop_loss_percent)
            else:
                return current_price >= self.entry_price * (1 + self.stop_loss_percent)
        
        return False
    
    def check_take_profit(self, current_price: float) -> bool:
        """检查是否触发止盈"""
        if self.take_profit_points is not None:
            if self.side == OrderSide.LONG:
                return current_price >= self.entry_price + self.take_profit_points
            else:
                return current_price <= self.entry_price - self.take_profit_points
        
        if self.take_profit_percent is not None:
            if self.side == OrderSide.LONG:
                return current_price >= self.entry_price * (1 + self.take_profit_percent)
            else:
                return current_price <= self.entry_price * (1 - self.take_profit_percent)
        
        return False
    
    def check_trailing_stop(self, current_price: float) -> bool:
        """检查是否触发移动止损"""
        if self.trailing_stop_pct is None:
            return False
        
        if self.side == OrderSide.LONG:
            trailing_stop_price = self.highest_price * (1 - self.trailing_stop_pct)
            return current_price <= trailing_stop_price
        else:  # SHORT
            trailing_stop_price = self.lowest_price * (1 + self.trailing_stop_pct)
            return current_price >= trailing_stop_price


@dataclass
class Order:
    """订单对象"""
    timestamp: pd.Timestamp
    side: OrderSide
    price: float
    quantity: int  # 手数
    contract: ContractConfig
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_timestamp: Optional[pd.Timestamp] = None
    order_id: Optional[int] = None
    slippage: float = 0.0  # 滑点（跳数）


@dataclass
class Trade:
    """交易记录"""
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    side: OrderSide
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float  # 绝对盈亏（元）
    pnl_pct: float  # 收益率
    commission: float  # 总手续费
    contract: ContractConfig
    exit_reason: str = "signal"  # signal, stop_loss, take_profit, trailing_stop, margin_call, end_of_backtest


class FutureAccount:
    """期货账户"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 contract: ContractConfig = None,
                 slippage: float = 0.1,  # 默认滑点0.1跳
                 max_drawdown: Optional[float] = None,  # 单日最大回撤
                 margin_call_level: float = 0.0):  # 强平线（可用保证金<=此值时强平）
        """
        参数:
        initial_capital: 初始本金
        contract: 合约配置
        slippage: 滑点（跳数）
        max_drawdown: 单日最大回撤百分比
        margin_call_level: 强平线（可用保证金<=此值时强平）
        """
        self.initial_capital = initial_capital
        self.available_margin = initial_capital  # 可用保证金
        self.used_margin = 0.0  # 已用保证金
        self.contract = contract or ContractConfig("DEFAULT")
        self.slippage = slippage
        self.max_drawdown = max_drawdown
        self.margin_call_level = margin_call_level
        
        self.position: Optional[FuturePosition] = None
        self.pending_orders: List[Order] = []
        self.trades: List[Trade] = []
        
        # 性能跟踪
        self.equity_curve = []
        self.timestamps = []
        self.daily_equity = []  # 每日权益
        self.daily_dates = []
        
        self._order_counter = 0
        self._daily_high_equity = initial_capital  # 当日最高权益（用于计算单日回撤）
        self._current_date = None
    
    def get_equity(self, current_price: float) -> float:
        """计算当前权益 = 可用保证金 + 已用保证金 + 浮动盈亏"""
        unrealized_pnl = 0.0
        if self.position is not None:
            unrealized_pnl = self.position.get_unrealized_pnl(current_price)
        
        return self.available_margin + self.used_margin + unrealized_pnl
    
    def get_available_margin(self, current_price: float) -> float:
        """计算可用保证金 = 权益 - 已用保证金"""
        equity = self.get_equity(current_price)
        return equity - self.used_margin
    
    def calculate_commission(self, price: float, quantity: int, is_open: bool, 
                            is_close_today: bool = False) -> float:
        """计算手续费"""
        contract_value = price * self.contract.multiplier * quantity
        
        # 判断是否使用平今仓费率
        if is_close_today and not is_open:
            if self.contract.close_today_commission_rate > 0:
                commission = contract_value * self.contract.close_today_commission_rate
            else:
                commission = contract_value * self.contract.commission_rate
            
            if self.contract.close_today_commission_per_contract > 0:
                commission += self.contract.close_today_commission_per_contract * quantity
            elif self.contract.commission_per_contract > 0:
                commission += self.contract.commission_per_contract * quantity
        else:
            # 普通开仓或平仓
            commission = contract_value * self.contract.commission_rate
            if self.contract.commission_per_contract > 0:
                commission += self.contract.commission_per_contract * quantity
        
        return commission
    
    def apply_slippage(self, price: float, side: OrderSide) -> float:
        """应用滑点"""
        tick_value = self.contract.price_tick_value / self.contract.multiplier  # 每跳价格
        slippage_price = self.slippage * tick_value
        
        if side == OrderSide.LONG:
            return price + slippage_price  # 做多时滑点向上
        elif side == OrderSide.SHORT:
            return price - slippage_price  # 做空时滑点向下
        else:  # CLOSE
            # 平仓时，根据持仓方向确定滑点
            if self.position and self.position.side == OrderSide.LONG:
                return price - slippage_price  # 平多时滑点向下
            else:
                return price + slippage_price  # 平空时滑点向上
    
    def can_open_position(self, price: float, quantity: int, side: OrderSide) -> Tuple[bool, str]:
        """检查是否可以开仓"""
        # 检查持仓限额
        if self.position is not None:
            if self.position.side == side:
                total_quantity = self.position.quantity + quantity
            else:
                # 反向开仓，需要先平仓
                return False, "已有反向持仓，需要先平仓"
        else:
            total_quantity = quantity
        
        if total_quantity > self.contract.max_position:
            return False, f"超过持仓限额 {self.contract.max_position} 手"
        
        # 计算所需保证金
        margin_required = price * self.contract.multiplier * quantity * self.contract.margin_ratio
        
        # 计算手续费
        commission = self.calculate_commission(price, quantity, is_open=True)
        
        # 检查可用保证金
        available = self.get_available_margin(price)
        if available < margin_required + commission:
            return False, f"可用保证金不足: 需要 {margin_required + commission:.2f}, 可用 {available:.2f}"
        
        return True, ""
    
    def create_order(self, timestamp: pd.Timestamp, side: OrderSide, 
                    price: float, quantity: int) -> Order:
        """创建订单"""
        order = Order(
            timestamp=timestamp,
            side=side,
            price=price,
            quantity=quantity,
            contract=self.contract,
            order_id=self._order_counter,
            slippage=self.slippage
        )
        self._order_counter += 1
        self.pending_orders.append(order)
        return order
    
    def fill_order(self, order: Order, fill_price: float, fill_timestamp: pd.Timestamp) -> bool:
        """执行订单，返回是否成功"""
        # 应用滑点
        actual_fill_price = self.apply_slippage(fill_price, order.side)
        
        if order.side == OrderSide.CLOSE:
            # 平仓
            if self.position is None:
                order.status = OrderStatus.REJECTED
                return False
            
            # 判断是否平今仓
            is_close_today = self.position.is_today_position
            
            # 计算手续费
            commission = self.calculate_commission(
                actual_fill_price, order.quantity, is_open=False, 
                is_close_today=is_close_today
            )
            
            # 计算盈亏
            pnl = self.position.get_unrealized_pnl(actual_fill_price)
            pnl_pct = pnl / (self.position.entry_price * self.contract.multiplier * self.position.quantity)
            
            # 计算开仓手续费
            open_commission = self.calculate_commission(
                self.position.entry_price, self.position.quantity, is_open=True
            )
            
            total_commission = commission + open_commission
            net_pnl = pnl - total_commission
            
            # 更新账户
            self.available_margin += self.used_margin + net_pnl
            self.used_margin = 0.0

            # 记录交易
            trade = Trade(
                entry_timestamp=self.position.entry_timestamp,
                exit_timestamp=fill_timestamp,
                side=self.position.side,
                entry_price=self.position.entry_price,
                exit_price=actual_fill_price,
                quantity=self.position.quantity,
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                commission=total_commission,
                exit_reason="signal",
                contract=self.contract
            )
            self.trades.append(trade)
            
            # 清空持仓
            self.position = None
            
        else:
            # 开仓
            # 检查是否可以开仓
            can_open, reason = self.can_open_position(actual_fill_price, order.quantity, order.side)
            if not can_open:
                order.status = OrderStatus.REJECTED
                return False
            
            # 如果已有同向持仓，加仓
            if self.position is not None and self.position.side == order.side:
                # 计算加权平均开仓价
                total_value = (self.position.entry_price * self.position.quantity + 
                              actual_fill_price * order.quantity)
                total_quantity = self.position.quantity + order.quantity
                avg_price = total_value / total_quantity
                
                self.position.entry_price = avg_price
                self.position.quantity = total_quantity
            else:
                # 反向开仓：如果有反向持仓，先平仓再开新仓
                # 例如：持有多仓时收到开空订单 → 先平多仓，再开空仓
                # 例如：持有空仓时收到开多订单 → 先平空仓，再开多仓
                if self.position is not None:
                    # 第一步：平掉现有反向持仓
                    # 平仓手续费
                    close_commission = self.calculate_commission(
                        actual_fill_price, self.position.quantity, is_open=False,
                        is_close_today=self.position.is_today_position
                    )
                    # 平仓盈亏
                    close_pnl = self.position.get_unrealized_pnl(actual_fill_price)
                    close_pnl_pct = close_pnl / (self.position.entry_price * self.contract.multiplier * self.position.quantity)
                    open_commission = self.calculate_commission(
                        self.position.entry_price, self.position.quantity, is_open=True
                    )
                    total_close_commission = close_commission + open_commission
                    net_close_pnl = close_pnl - total_close_commission
                    
                    # 记录平仓交易（exit_reason="reverse" 表示反向开仓导致的平仓）
                    trade = Trade(
                        entry_timestamp=self.position.entry_timestamp,
                        exit_timestamp=fill_timestamp,
                        side=self.position.side,
                        entry_price=self.position.entry_price,
                        exit_price=actual_fill_price,
                        quantity=self.position.quantity,
                        pnl=net_close_pnl,
                        pnl_pct=close_pnl_pct,
                        commission=total_close_commission,
                        exit_reason="reverse",
                        contract=self.contract
                    )
                    self.trades.append(trade)
                    
                    # 更新账户（释放保证金并加上盈亏）
                    self.available_margin += self.used_margin + net_close_pnl
                    self.used_margin = 0.0
                
                # 第二步：开新仓
                # 计算所需保证金和手续费
                margin_required = actual_fill_price * self.contract.multiplier * order.quantity * self.contract.margin_ratio
                commission = self.calculate_commission(actual_fill_price, order.quantity, is_open=True)
                
                # 更新账户
                self.used_margin = margin_required
                self.available_margin -= (margin_required + commission)
                
                # 创建持仓
                self.position = FuturePosition(
                    side=order.side,
                    entry_price=actual_fill_price,
                    entry_timestamp=fill_timestamp,
                    quantity=order.quantity,
                    contract=self.contract,
                    is_today_position=True  # 新开仓默认为今日持仓
                )
        
        order.status = OrderStatus.FILLED
        order.fill_price = actual_fill_price
        order.fill_timestamp = fill_timestamp
        
        # 从待处理订单中移除
        if order in self.pending_orders:
            self.pending_orders.remove(order)
        
        return True
    
    def update_position(self, current_price: float, timestamp: pd.Timestamp) -> Optional[str]:
        """更新持仓状态，检查止盈止损和强平
        
        注意：如果所有风控参数（stop_loss_*, take_profit_*, trailing_stop_pct, max_drawdown）均为None，
        且margin_call_level=0，则只会根据signal字段交易，不会触发其他平仓机制。
        """
        if self.position is None:
            return None
        
        # 更新价格
        self.position.update_price(current_price)
        
        # 更新当日最高权益（用于计算单日回撤）
        current_date = timestamp.date() if hasattr(timestamp, 'date') else pd.Timestamp(timestamp).date()
        if self._current_date != current_date:
            self._current_date = current_date
            self._daily_high_equity = self.get_equity(current_price)
        
        current_equity = self.get_equity(current_price)
        self._daily_high_equity = max(self._daily_high_equity, current_equity)
        
        # 检查单日最大回撤（只有当max_drawdown不为None时才检查）
        if self.max_drawdown is not None:
            daily_dd = (self._daily_high_equity - current_equity) / self._daily_high_equity
            if daily_dd >= self.max_drawdown:
                # 触发单日最大回撤，强平
                commission = self.calculate_commission(
                    current_price, self.position.quantity, is_open=False,
                    is_close_today=self.position.is_today_position
                )
                self._force_close_position(current_price, timestamp, commission, "max_drawdown")
                return "max_drawdown"
        
        # 检查可用保证金（强平）- 只有当可用保证金 <= margin_call_level 时才触发
        available = self.get_available_margin(current_price)
        if available <= self.margin_call_level:
            commission = self.calculate_commission(
                current_price, self.position.quantity, is_open=False,
                is_close_today=self.position.is_today_position
            )
            self._force_close_position(current_price, timestamp, commission, "margin_call")
            return "margin_call"
        
        # 检查移动止损（只有当trailing_stop_pct不为None时，check_trailing_stop才会返回True）
        if self.position.check_trailing_stop(current_price):
            commission = self.calculate_commission(
                current_price, self.position.quantity, is_open=False,
                is_close_today=self.position.is_today_position
            )
            self._force_close_position(current_price, timestamp, commission, "trailing_stop")
            return "trailing_stop"
        
        # 检查止损（只有当stop_loss_points或stop_loss_percent不为None时，check_stop_loss才会返回True）
        if self.position.check_stop_loss(current_price):
            commission = self.calculate_commission(
                current_price, self.position.quantity, is_open=False,
                is_close_today=self.position.is_today_position
            )
            self._force_close_position(current_price, timestamp, commission, "stop_loss")
            return "stop_loss"
        
        # 检查止盈（只有当take_profit_points或take_profit_percent不为None时，check_take_profit才会返回True）
        if self.position.check_take_profit(current_price):
            commission = self.calculate_commission(
                current_price, self.position.quantity, is_open=False,
                is_close_today=self.position.is_today_position
            )
            self._force_close_position(current_price, timestamp, commission, "take_profit")
            return "take_profit"
        
        return None
    
    def _force_close_position(self, exit_price: float, exit_timestamp: pd.Timestamp,
                              commission: float, exit_reason: str):
        """强制平仓"""
        if self.position is None:
            return
        
        # 计算盈亏
        pnl = self.position.get_unrealized_pnl(exit_price)
        pnl_pct = pnl / (self.position.entry_price * self.contract.multiplier * self.position.quantity)
        
        # 计算开仓手续费
        open_commission = self.calculate_commission(
            self.position.entry_price, self.position.quantity, is_open=True
        )
        
        total_commission = commission + open_commission
        net_pnl = pnl - total_commission
        
        # 更新账户
        self.available_margin += self.used_margin + net_pnl
        self.used_margin = 0.0
        
        # 记录交易
        trade = Trade(
            entry_timestamp=self.position.entry_timestamp,
            exit_timestamp=exit_timestamp,
            side=self.position.side,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            quantity=self.position.quantity,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=total_commission,
            exit_reason=exit_reason,
            contract=self.contract
        )
        self.trades.append(trade)
        
        # 清空持仓
        self.position = None

    def record_equity(self, timestamp: pd.Timestamp, current_price: float):
        """记录权益曲线"""
        equity = self.get_equity(current_price)
        self.equity_curve.append(equity)
        self.timestamps.append(timestamp)
        
        # 记录每日权益（用于日度统计）
        current_date = timestamp.date() if hasattr(timestamp, 'date') else pd.Timestamp(timestamp).date()
        if not self.daily_dates or self.daily_dates[-1] != current_date:
            self.daily_equity.append(equity)
            self.daily_dates.append(current_date)
        else:
            # 更新当日权益
            self.daily_equity[-1] = equity
    
    def get_metrics(self, risk_free_rate: float = 0.0) -> Dict:
        """计算回测指标"""
        if len(self.equity_curve) == 0:
            return {"error": "无回测数据"}
        
        equity_array = np.array(self.equity_curve)
        final_equity = equity_array[-1]
        
        # 基本指标
        total_pnl = final_equity - self.initial_capital
        total_return = total_pnl / self.initial_capital
        
        # 年化收益率
        if len(self.timestamps) > 1:
            days = (self.timestamps[-1] - self.timestamps[0]).days
            if days > 0:
                annual_return = (final_equity / self.initial_capital) ** (365.0 / days) - 1
            else:
                annual_return = 0.0
        else:
            annual_return = 0.0
        
        # 最大回撤
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        # 卡玛比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # 收益率序列（用于计算夏普和索提诺）
        returns = np.diff(equity_array) / equity_array[:-1]
        returns = returns[~np.isnan(returns)]
        
        # 夏普比率
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) - risk_free_rate / 252) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 索提诺比率（只考虑下行波动）
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and np.std(downside_returns) > 0:
            sortino_ratio = (np.mean(returns) - risk_free_rate / 252) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino_ratio = 0.0
        
        # 交易统计
        if len(self.trades) == 0:
            return {
                "total_pnl": total_pnl,
                "total_return": total_return,
                "annual_return": annual_return,
                "max_drawdown": max_drawdown,
                "calmar_ratio": calmar_ratio,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "trade_count": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_pnl_per_trade": 0.0,
                "final_equity": final_equity
            }
        
        # 胜率
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
        
        # 盈利因子
        total_profit = sum([t.pnl for t in winning_trades]) if winning_trades else 0.0
        total_loss = abs(sum([t.pnl for t in losing_trades])) if losing_trades else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        # 平均每笔盈亏
        avg_pnl_per_trade = np.mean([t.pnl for t in self.trades]) if self.trades else 0.0
        
        # 平均盈利/亏损
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        return {
            "total_pnl": total_pnl,
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "trade_count": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_pnl_per_trade": avg_pnl_per_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_commission": sum([t.commission for t in self.trades]),
            "final_equity": final_equity,
            "initial_capital": self.initial_capital
        }


class Strategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str = "Strategy"):
        self.name = name
        self.account: Optional[FutureAccount] = None
        self.data: Optional[pd.DataFrame] = None
    
    def initialize(self, account: FutureAccount, data: pd.DataFrame):
        """初始化策略"""
        self.account = account
        self.data = data
        self.prepare_data()
    
    def prepare_data(self):
        """准备数据，计算指标（子类可重写）"""
        pass
    
    @abstractmethod
    def on_bar(self, bar: pd.Series, index: int) -> Optional[Order]:
        """
        每个Bar的处理逻辑
        
        参数:
        bar: 当前Bar的数据
        index: 当前Bar的索引
        
        返回:
        Order对象或None
        """
        pass
    
    def on_order_filled(self, order: Order):
        """订单成交回调（可选实现）"""
        pass
    
    def on_position_closed(self, trade: Trade):
        """持仓平仓回调（可选实现）"""
        pass
    
    def on_stop_loss(self, trade: Trade):
        """止损触发回调（可选实现）"""
        pass
    
    def on_take_profit(self, trade: Trade):
        """止盈触发回调（可选实现）"""
        pass
    
    def on_margin_call(self, trade: Trade):
        """强平触发回调（可选实现）"""
        pass


class SignalStrategy(Strategy):
    """基于信号的策略"""
    
    def __init__(self, 
                 name: str = "SignalStrategy",
                 signal_column: str = "signal",
                 quantity: int = 1,
                 stop_loss_points: Optional[float] = None,
                 stop_loss_percent: Optional[float] = None,
                 take_profit_points: Optional[float] = None,
                 take_profit_percent: Optional[float] = None,
                 trailing_stop_pct: Optional[float] = None):
        """
        参数:
        signal_column: 信号列名（1=做多, -1=做空, 0=平仓）
        quantity: 每次交易手数
        stop_loss_points: 固定止损点数
        stop_loss_percent: 百分比止损
        take_profit_points: 固定止盈点数
        take_profit_percent: 百分比止盈
        trailing_stop_pct: 移动止损百分比
        """
        super().__init__(name)
        self.signal_column = signal_column
        self.quantity = quantity
        self.stop_loss_points = stop_loss_points
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_points = take_profit_points
        self.take_profit_percent = take_profit_percent
        self.trailing_stop_pct = trailing_stop_pct
        self.last_signal: Optional[int] = None  # 跟踪上一个信号
        self.last_position_side: Optional[OrderSide] = None  # 跟踪上一个bar的持仓方向（用于检测持仓状态变化）
    
    def prepare_data(self):
        """准备数据，重置上一个信号和持仓状态"""
        super().prepare_data()
        self.last_signal = None  # 每次回测开始时重置
        self.last_position_side = None  # 重置持仓状态跟踪

    def on_bar(self, bar: pd.Series, index: int):
        signal = int(bar[self.signal_column])
        price = bar['close']  # 强制使用 close
        timestamp = bar.name  # 当前时间
        symbol = bar['symbol']

        pos = self.account.position
        # 获取持仓方向：None=无仓, OrderSide.LONG=1, OrderSide.SHORT=-1
        current_side = pos.side.value if pos else 0

        # === 反向信号：立即平仓 + 立即开仓（同价、同时间）===
        if current_side != 0 and signal != 0 and signal != current_side:
            # 平仓
            self.close_position(
                exit_time=timestamp,
                exit_price=price,
                exit_reason="signal",
                symbol=symbol
            )
            # 开仓
            new_side = "LONG" if signal == 1 else "SHORT"
            self.open_position(
                entry_time=timestamp,
                side=new_side,
                entry_price=price,
                quantity=1,
                symbol=symbol
            )
            return None

        # === 开仓：无仓 + 有信号 ===
        if current_side == 0 and signal in [1, -1]:
            side = "LONG" if signal == 1 else "SHORT"
            self.open_position(
                entry_time=timestamp,
                side=side,
                entry_price=price,
                quantity=1,
                symbol=symbol
            )
            return None

        # === 平仓：有仓 + 信号=0 ===
        if current_side != 0 and signal == 0:
            self.close_position(
                exit_time=timestamp,
                exit_price=price,
                exit_reason="signal",
                symbol=symbol
            )
            return None
        
        # 其他情况：无操作
        return None

    def open_position(self, entry_time, side, entry_price, quantity, symbol):
        """开仓（便捷方法）"""
        # 将字符串转换为 OrderSide 枚举
        if side == "LONG":
            order_side = OrderSide.LONG
        elif side == "SHORT":
            order_side = OrderSide.SHORT
        else:
            raise ValueError(f"无效的方向: {side}，必须是 'LONG' 或 'SHORT'")
        
        # 创建开仓订单
        order = self.account.create_order(entry_time, order_side, entry_price, quantity)
        
        # 立即执行订单（回测中假设立即成交）
        success = self.account.fill_order(order, entry_price, entry_time)
        if success:
            self.on_order_filled(order)

    def close_position(self, exit_time, exit_price, exit_reason, symbol):
        """平仓（便捷方法）"""
        if self.account.position is None:
            return  # 无持仓，无需平仓
        
        # 创建平仓订单
        order = self.account.create_order(
            exit_time, 
            OrderSide.CLOSE, 
            exit_price, 
            self.account.position.quantity
        )
        
        # 立即执行订单（回测中假设立即成交）
        success = self.account.fill_order(order, exit_price, exit_time)
        if success:
            self.on_order_filled(order)
            
            # 如果平仓成功，更新交易记录的exit_reason
            if len(self.account.trades) > 0:
                self.account.trades[-1].exit_reason = exit_reason
    def on_order_filled(self, order: Order):
        """订单成交后设置止盈止损，并更新持仓状态跟踪
        
        注意：如果所有止盈止损参数均为None，则只根据signal字段交易，不会触发止盈止损平仓
        """
        # 更新持仓状态跟踪（订单成交后的实际持仓状态）
        if self.account.position is not None:
            self.last_position_side = self.account.position.side
        else:
            # 平仓后无持仓
            self.last_position_side = None
        
        if order.side != OrderSide.CLOSE and self.account.position is not None:
            position = self.account.position
            
            # 设置止损（如果参数为None，check_stop_loss会返回False，不会触发平仓）
            position.stop_loss_points = self.stop_loss_points
            position.stop_loss_percent = self.stop_loss_percent
            
            # 设置止盈（如果参数为None，check_take_profit会返回False，不会触发平仓）
            position.take_profit_points = self.take_profit_points
            position.take_profit_percent = self.take_profit_percent
            
            # 设置移动止损（如果参数为None，check_trailing_stop会返回False，不会触发平仓）
            position.trailing_stop_pct = self.trailing_stop_pct
    
    def on_position_closed(self, trade: Trade):
        """持仓平仓回调：更新持仓状态跟踪"""
        # 平仓后无持仓，更新状态跟踪
        self.last_position_side = None
    
    def on_stop_loss(self, trade: Trade):
        """止损触发回调：更新持仓状态跟踪"""
        # 平仓后无持仓，更新状态跟踪
        self.last_position_side = None
    
    def on_take_profit(self, trade: Trade):
        """止盈触发回调：更新持仓状态跟踪"""
        # 平仓后无持仓，更新状态跟踪
        self.last_position_side = None
    
    def on_margin_call(self, trade: Trade):
        """强平触发回调：更新持仓状态跟踪"""
        # 平仓后无持仓，更新状态跟踪
        self.last_position_side = None


class FutureBacktest:
    """期货回测引擎"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 strategy: Strategy,
                 contract: ContractConfig,
                 initial_capital: float = 100000.0,
                 slippage: float = 0.1,
                 max_drawdown: Optional[float] = None,
                 margin_call_level: float = 0.0):
        """
        参数:
        data: 包含价格和信号的DataFrame
        strategy: 策略对象
        contract: 合约配置
        initial_capital: 初始资金
        slippage: 滑点（跳数）
        max_drawdown: 单日最大回撤百分比
        margin_call_level: 强平线
        """
        self.data = data.copy()
        self.strategy = strategy
        self.contract = contract
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.max_drawdown = max_drawdown
        self.margin_call_level = margin_call_level
        
        self.account = FutureAccount(
            initial_capital=initial_capital,
            contract=contract,
            slippage=slippage,
            max_drawdown=max_drawdown,
            margin_call_level=margin_call_level
        )
        
        # 确保数据有date列且为索引
        if 'date' not in self.data.columns and not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("数据必须包含'date'列或DatetimeIndex")
        
        if 'date' in self.data.columns and not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.set_index('date', inplace=True, drop=False)
    
    def run(self, verbose: bool = True) -> Dict:
        """运行回测"""
        # 初始化策略
        self.strategy.initialize(self.account, self.data)
        
        if verbose:
            print(f"开始回测: {self.strategy.name}")
            print(f"合约: {self.contract.symbol}")
            print(f"数据范围: {self.data.index[0]} 到 {self.data.index[-1]}")
            print(f"数据点数: {len(self.data)}")
            print(f"初始资金: {self.initial_capital:.2f}")
            print(f"保证金比例: {self.contract.margin_ratio*100:.1f}%")
            print(f"手续费率: {self.contract.commission_rate*10000:.2f} 万")
            print(f"滑点: {self.slippage} 跳")
            print("-" * 50)
        
        # 事件循环
        for idx, (timestamp, bar) in enumerate(self.data.iterrows()):
            current_price = bar['close']
            
            # 更新持仓状态，检查止盈止损和强平
            if self.account.position is not None:
                exit_reason = self.account.update_position(current_price, timestamp)
                
                if exit_reason and len(self.account.trades) > 0:
                    last_trade = self.account.trades[-1]
                    
                    if exit_reason == "stop_loss":
                        self.strategy.on_stop_loss(last_trade)
                    elif exit_reason == "take_profit":
                        self.strategy.on_take_profit(last_trade)
                    elif exit_reason == "margin_call":
                        self.strategy.on_margin_call(last_trade)
                    
                    self.strategy.on_position_closed(last_trade)
            
            # 处理待执行订单
            for order in self.account.pending_orders[:]:
                success = self.account.fill_order(order, current_price, timestamp)
                if success:
                    self.strategy.on_order_filled(order)
            
            # 策略生成新订单
            order = self.strategy.on_bar(bar, idx)
            if order is not None:
                # 立即尝试成交（回测中假设立即成交）
                success = self.account.fill_order(order, current_price, timestamp)
                if success:
                    self.strategy.on_order_filled(order)
            
            # 记录权益
            self.account.record_equity(timestamp, current_price)
        
        # 回测结束时强平
        if self.account.position is not None:
            last_bar = self.data.iloc[-1]
            last_price = last_bar['close']
            last_timestamp = self.data.index[-1]
            commission = self.account.calculate_commission(
                last_price, self.account.position.quantity, is_open=False,
                is_close_today=self.account.position.is_today_position
            )
            self.account._force_close_position(last_price, last_timestamp, commission, "end_of_backtest")
        
        # 计算指标
        metrics = self.account.get_metrics()
        
        if verbose:
            print("\n回测结果:")
            print("-" * 50)
            for key, value in metrics.items():
                if isinstance(value, float):
                    if abs(value) < 0.01:
                        print(f"{key}: {value:.6f}")
                    else:
                        print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        return metrics
    
    def get_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        获取回测结果
        
        返回:
        trades_df: 交易记录DataFrame
        equity_df: 权益曲线DataFrame
        metrics: 性能指标字典
        """
        # 交易记录
        trades_data = []
        for trade in self.account.trades:
            # 统一格式化时间戳为 YYYY-MM-DD HH:MM:SS
            entry_time_str = pd.Timestamp(trade.entry_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            exit_time_str = pd.Timestamp(trade.exit_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            trades_data.append({
                'entry_time': entry_time_str,
                'exit_time': exit_time_str,
                'side': trade.side.name,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'commission': trade.commission,
                'exit_reason': trade.exit_reason
            })
        trades_df = pd.DataFrame(trades_data)
        
        # 权益曲线
        equity_df = pd.DataFrame({
            'timestamp': self.account.timestamps,
            'equity': self.account.equity_curve
        })
        equity_df.set_index('timestamp', inplace=True)
        
        # 性能指标
        metrics = self.account.get_metrics()
        
        return trades_df, equity_df, metrics


class MultiSymbolBacktest:
    """多品种期货回测引擎
    
    支持根据 df_signals (包含 symbol, date, close, signal) 进行多品种回测
    """
    
    def __init__(self,
                 df_signals: pd.DataFrame,
                 contract_config_func: Optional[Callable[[str], ContractConfig]] = None,
                 initial_capital: float = 100000.0,
                 capital_per_symbol: Optional[float] = None,
                 slippage: float = 0.1,
                 max_drawdown: Optional[float] = None,
                 margin_call_level: float = 0.0,
                 quantity: int = 1,
                 stop_loss_points: Optional[float] = None,
                 stop_loss_percent: Optional[float] = None,
                 take_profit_points: Optional[float] = None,
                 take_profit_percent: Optional[float] = None,
                 trailing_stop_pct: Optional[float] = None):
        """
        参数:
        df_signals: 包含 'symbol', 'date', 'close', 'signal' 列的DataFrame
        contract_config_func: 根据symbol返回ContractConfig的函数，如果为None则使用默认配置
        initial_capital: 总初始资金
        capital_per_symbol: 每个品种分配的资金（如果为None，则平均分配）
        slippage: 滑点（跳数）
        max_drawdown: 单日最大回撤百分比
        margin_call_level: 强平线
        quantity: 每次交易手数
        stop_loss_points: 固定止损点数
        stop_loss_percent: 百分比止损
        take_profit_points: 固定止盈点数
        take_profit_percent: 百分比止盈
        trailing_stop_pct: 移动止损百分比
        """
        # 验证数据格式
        required_cols = ['symbol', 'date', 'close', 'signal']
        if not all(col in df_signals.columns for col in required_cols):
            raise ValueError(f"df_signals 必须包含以下列: {required_cols}")
        
        self.df_signals = df_signals.copy()
        
        # 确保 date 列是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(self.df_signals['date']):
            self.df_signals['date'] = pd.to_datetime(self.df_signals['date'])
        
        # 按日期和品种排序
        self.df_signals = self.df_signals.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        # 获取所有品种
        self.symbols = sorted(self.df_signals['symbol'].unique())
        self.num_symbols = len(self.symbols)
        
        # 计算每个品种的资金
        if capital_per_symbol is None:
            self.capital_per_symbol = initial_capital / self.num_symbols if self.num_symbols > 0 else initial_capital
        else:
            self.capital_per_symbol = capital_per_symbol
        
        # 合约配置函数
        if contract_config_func is None:
            def default_contract_config(symbol: str) -> ContractConfig:
                """默认合约配置（可根据实际需求修改）"""
                return ContractConfig(
                    symbol=symbol,
                    multiplier=10.0,
                    tick_size=1.0,
                    margin_ratio=0.12,
                    commission_rate=0.00023
                )
            self.contract_config_func = default_contract_config
        else:
            self.contract_config_func = contract_config_func
        
        # 为每个品种创建回测引擎
        self.backtests: Dict[str, FutureBacktest] = {}
        self.symbol_metrics: Dict[str, Dict] = {}
        
        # 策略参数
        self.slippage = slippage
        self.max_drawdown = max_drawdown
        self.margin_call_level = margin_call_level
        self.quantity = quantity
        self.stop_loss_points = stop_loss_points
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_points = take_profit_points
        self.take_profit_percent = take_profit_percent
        self.trailing_stop_pct = trailing_stop_pct
        
        # 初始化每个品种的回测
        self._initialize_backtests()
    
    def _initialize_backtests(self):
        """为每个品种初始化回测引擎"""
        for symbol in self.symbols:
            # 获取该品种的数据
            symbol_data = self.df_signals[self.df_signals['symbol'] == symbol].copy()
            
            if len(symbol_data) == 0:
                continue
            
            # 确保 date 列是 datetime 类型
            if 'date' in symbol_data.columns:
                if not pd.api.types.is_datetime64_any_dtype(symbol_data['date']):
                    symbol_data['date'] = pd.to_datetime(symbol_data['date'])
            
            # 设置索引为 date，并确保索引是 DatetimeIndex
            symbol_data = symbol_data.set_index('date')
            if not isinstance(symbol_data.index, pd.DatetimeIndex):
                symbol_data.index = pd.to_datetime(symbol_data.index)
            
            # 按时间排序
            symbol_data = symbol_data.sort_index()
            
            # 确保有必要的列，特别是close列（必须有才能交易）
            if 'close' not in symbol_data.columns:
                raise ValueError(f"品种 {symbol} 的数据缺少 'close' 列")
            
            # 过滤掉close为NaN的行（这些行无法交易）
            before_filter = len(symbol_data)
            symbol_data = symbol_data[symbol_data['close'].notna()].copy()
            after_filter = len(symbol_data)
            
            if before_filter > after_filter:
                print(f"警告: 品种 {symbol} 过滤掉了 {before_filter - after_filter} 行close为NaN的数据")
            
            if len(symbol_data) == 0:
                print(f"警告: 品种 {symbol} 过滤后没有有效数据，跳过")
                continue
            
            # 确保有必要的列
            if 'open' not in symbol_data.columns:
                symbol_data['open'] = symbol_data['close']
            if 'high' not in symbol_data.columns:
                symbol_data['high'] = symbol_data['close'] * 1.001
            if 'low' not in symbol_data.columns:
                symbol_data['low'] = symbol_data['close'] * 0.999
            if 'volume' not in symbol_data.columns:
                symbol_data['volume'] = 0
            
            # 获取合约配置
            contract = self.contract_config_func(symbol)
            
            # 创建策略
            strategy = SignalStrategy(
                name=f"信号策略_{symbol}",
                signal_column="signal",
                quantity=self.quantity,
                stop_loss_points=self.stop_loss_points,
                stop_loss_percent=self.stop_loss_percent,
                take_profit_points=self.take_profit_points,
                take_profit_percent=self.take_profit_percent,
                trailing_stop_pct=self.trailing_stop_pct
            )
            
            # 创建回测引擎
            backtest = FutureBacktest(
                data=symbol_data,
                strategy=strategy,
                contract=contract,
                initial_capital=self.capital_per_symbol,
                slippage=self.slippage,
                max_drawdown=self.max_drawdown,
                margin_call_level=self.margin_call_level
            )
            
            self.backtests[symbol] = backtest
    
    def run(self, verbose: bool = True) -> Dict:
        """运行多品种回测"""
        if verbose:
            print("=" * 60)
            print("多品种期货回测")
            print("=" * 60)
            print(f"品种数量: {self.num_symbols}")
            print(f"品种列表: {', '.join(self.symbols)}")
            print(f"每个品种初始资金: {self.capital_per_symbol:.2f}")
            print(f"总初始资金: {self.capital_per_symbol * self.num_symbols:.2f}")
            print(f"数据范围: {self.df_signals['date'].min()} 到 {self.df_signals['date'].max()}")
            print("-" * 60)
        
        # 运行每个品种的回测
        all_trades = []
        all_equity_curves = []
        total_initial_capital = 0
        total_final_equity = 0
        
        for symbol in self.symbols:
            if symbol not in self.backtests:
                continue
            
            if verbose:
                print(f"\n回测品种: {symbol}")
                print("-" * 60)
            
            backtest = self.backtests[symbol]
            metrics = backtest.run(verbose=verbose)
            self.symbol_metrics[symbol] = metrics
            
            # 获取交易记录和权益曲线
            trades_df, equity_df, _ = backtest.get_results()
            
            # 添加品种列
            trades_df['symbol'] = symbol
            
            # 重置权益曲线的索引，确保timestamp作为列存在
            # equity_df 的索引是 'timestamp'，需要重置为列
            if equity_df.index.name == 'timestamp' or 'timestamp' not in equity_df.columns:
                equity_df = equity_df.reset_index()
            equity_df['symbol'] = symbol
            
            all_trades.append(trades_df)
            all_equity_curves.append(equity_df)
            
            total_initial_capital += backtest.initial_capital
            total_final_equity += metrics.get('final_equity', backtest.initial_capital)
        
        # 汇总结果
        if len(all_trades) > 0:
            self.all_trades_df = pd.concat(all_trades, ignore_index=True)
            self.all_equity_df = pd.concat(all_equity_curves, ignore_index=True)
        else:
            self.all_trades_df = pd.DataFrame()
            self.all_equity_df = pd.DataFrame()
        
        # 计算汇总指标
        summary_metrics = self._calculate_summary_metrics(
            total_initial_capital, total_final_equity
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("汇总回测结果")
            print("=" * 60)
            for key, value in summary_metrics.items():
                if isinstance(value, float):
                    if abs(value) < 0.01:
                        print(f"{key}: {value:.6f}")
                    else:
                        print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        return summary_metrics
    
    def _calculate_summary_metrics(self, total_initial: float, total_final: float) -> Dict:
        """计算汇总指标"""
        if total_initial == 0:
            return {"error": "无有效回测数据"}
        
        total_pnl = total_final - total_initial
        total_return = total_pnl / total_initial
        
        # 计算时间跨度
        if len(self.df_signals) > 0:
            start_date = self.df_signals['date'].min()
            end_date = self.df_signals['date'].max()
            days = (end_date - start_date).days
            if days > 0:
                annual_return = (total_final / total_initial) ** (365.0 / days) - 1
            else:
                annual_return = 0.0
        else:
            annual_return = 0.0
            days = 0
        
        # 汇总所有交易
        if len(self.all_trades_df) > 0:
            total_trades = len(self.all_trades_df)
            winning_trades = len(self.all_trades_df[self.all_trades_df['pnl'] > 0])
            losing_trades = len(self.all_trades_df[self.all_trades_df['pnl'] <= 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            total_profit = self.all_trades_df[self.all_trades_df['pnl'] > 0]['pnl'].sum()
            total_loss = abs(self.all_trades_df[self.all_trades_df['pnl'] <= 0]['pnl'].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
            
            avg_pnl = self.all_trades_df['pnl'].mean()
            total_commission = self.all_trades_df['commission'].sum()
        else:
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            win_rate = 0.0
            profit_factor = 0.0
            avg_pnl = 0.0
            total_commission = 0.0
        
        # 计算权益曲线回撤
        if len(self.all_equity_df) > 0:
            # 按时间汇总所有品种的权益
            equity_by_time = self.all_equity_df.groupby('timestamp')['equity'].sum().sort_index()
            equity_array = equity_by_time.values
            
            if len(equity_array) > 0:
                running_max = np.maximum.accumulate(equity_array)
                drawdown = (equity_array - running_max) / running_max
                max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
                
                # 计算夏普比率
                returns = np.diff(equity_array) / equity_array[:-1]
                returns = returns[~np.isnan(returns)]
                if len(returns) > 1 and np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
            else:
                max_drawdown = 0.0
                sharpe_ratio = 0.0
        else:
            max_drawdown = 0.0
            sharpe_ratio = 0.0
        
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        
        return {
            "total_initial_capital": total_initial,
            "total_final_equity": total_final,
            "total_pnl": total_pnl,
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "sharpe_ratio": sharpe_ratio,
            "trade_count": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_pnl_per_trade": avg_pnl,
            "total_commission": total_commission,
            "num_symbols": self.num_symbols,
            "backtest_days": days
        }
    
    def get_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict[str, Dict]]:
        """
        获取回测结果
        
        返回:
        all_trades_df: 所有品种的交易记录DataFrame
        all_equity_df: 所有品种的权益曲线DataFrame
        summary_metrics: 汇总性能指标字典
        symbol_metrics: 每个品种的性能指标字典
        """
        return self.all_trades_df, self.all_equity_df, self._calculate_summary_metrics(
            sum(bt.initial_capital for bt in self.backtests.values()),
            sum(m.get('final_equity', bt.initial_capital) 
                for bt, m in zip(self.backtests.values(), self.symbol_metrics.values()))
        ), self.symbol_metrics
    
    @staticmethod
    def check_backtest_data(
        df_signals: pd.DataFrame,
        df_trades: Optional[pd.DataFrame] = None,
        contract_config_func: Optional[Callable[[str], ContractConfig]] = None,
        initial_capital: float = 100000.0,
        slippage: float = 0.1,
        quantity: int = 1,
        stop_loss_points: Optional[float] = None,
        stop_loss_percent: Optional[float] = None,
        take_profit_points: Optional[float] = None,
        take_profit_percent: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        margin_call_level: float = 0.0,
        # 合约配置参数（用于默认合约配置函数）
        commission_per_contract: float = 0.0,
        close_today_commission_rate: float = 0.0,
        close_today_commission_per_contract: float = 0.0,
        max_position: int = 5000,
        trade_unit: int = 1,
        verbose: bool = True
    ) -> 'MultiSymbolBacktest':
        """
        检查回测数据并创建多品种回测对象
        
        参数:
        df_signals: 信号数据，必须包含 'symbol', 'date', 'signal' 列，如果包含 'close' 则不需要 df_trades
        df_trades: 价格数据，包含 'symbol', 'date', 'close' 列（可选，如果 df_signals 已包含 close 则不需要）
        contract_config_func: 合约配置函数，根据symbol返回ContractConfig（可选）
        initial_capital: 总初始资金
        slippage: 滑点（跳数）
        quantity: 每次交易手数
        stop_loss_points: 固定止损点数（与stop_loss_percent二选一）
        stop_loss_percent: 百分比止损（与stop_loss_points二选一）
        take_profit_points: 固定止盈点数（与take_profit_percent二选一）
        take_profit_percent: 百分比止盈（与take_profit_points二选一）
        trailing_stop_pct: 移动止损百分比
        max_drawdown: 单日最大回撤百分比
        margin_call_level: 强平线（可用保证金<=此值时强平）
        commission_per_contract: 固定手续费（按手，用于默认合约配置）
        close_today_commission_rate: 平今仓手续费率（用于默认合约配置）
        close_today_commission_per_contract: 平今仓固定手续费（用于默认合约配置）
        max_position: 单品种持仓限额（用于默认合约配置）
        trade_unit: 最小交易手数（用于默认合约配置）
        verbose: 是否打印详细信息
        
        返回:
        multi_backtest: MultiSymbolBacktest对象
        """
        # 确保 df_signals 包含必要的列
        df_signals = df_signals.copy()
        if 'symbol' not in df_signals.columns:
            df_signals['symbol'] = 'A2507'  # 如果没有symbol列，添加默认值
        if 'date' not in df_signals.columns:
            if isinstance(df_signals.index, pd.DatetimeIndex):
                df_signals['date'] = df_signals.index
            else:
                raise ValueError("数据必须包含'date'列或DatetimeIndex")
        
        # 确保 df_signals 包含 'close' 列（从 df_trades 合并或使用原始数据）
        if 'close' not in df_signals.columns:
            if df_trades is not None and 'close' in df_trades.columns:
                # 确保两个DataFrame的date列都是datetime类型
                df_signals_date = df_signals['date'].copy()
                df_trades_date = df_trades['date'].copy()
                
                if not pd.api.types.is_datetime64_any_dtype(df_signals_date):
                    df_signals_date = pd.to_datetime(df_signals_date)
                if not pd.api.types.is_datetime64_any_dtype(df_trades_date):
                    df_trades_date = pd.to_datetime(df_trades_date)
                
                # 从 df_trades 合并 close 价格
                # 先尝试精确匹配（on=['symbol', 'date']）
                df_signals_temp = df_signals.copy()
                df_signals_temp['date'] = df_signals_date
                
                df_trades_temp = df_trades[['symbol', 'date', 'close']].copy()
                df_trades_temp['date'] = df_trades_date
                
                # 先尝试精确匹配
                df_merged = df_signals_temp.merge(
                    df_trades_temp, 
                    on=['symbol', 'date'], 
                    how='left'
                )
                
                # 如果精确匹配后有NaN，尝试使用merge_asof进行最近邻匹配（1分钟内）
                if df_merged['close'].isna().any():
                    # 保存原始索引
                    df_merged['_orig_idx'] = df_merged.index
                    
                    # 为每个品种分别进行merge_asof
                    nan_mask = df_merged['close'].isna()
                    df_signals_nan = df_merged[nan_mask].copy()
                    
                    if len(df_signals_nan) > 0:
                        # 按品种分组处理
                        for symbol in df_signals_nan['symbol'].unique():
                            symbol_signals = df_signals_nan[df_signals_nan['symbol'] == symbol].sort_values('date')
                            symbol_trades = df_trades_temp[df_trades_temp['symbol'] == symbol].sort_values('date')
                            
                            if len(symbol_signals) > 0 and len(symbol_trades) > 0:
                                # 使用merge_asof进行最近邻匹配
                                df_asof = pd.merge_asof(
                                    symbol_signals[['date', '_orig_idx']],
                                    symbol_trades[['date', 'close']],
                                    on='date',
                                    direction='nearest',
                                    tolerance=pd.Timedelta(seconds=60)  # 允许1分钟内的匹配
                                )
                                
                                # 更新匹配到的close价格
                                for _, row in df_asof.iterrows():
                                    if pd.notna(row['close']):
                                        orig_idx = int(row['_orig_idx'])
                                        df_merged.loc[orig_idx, 'close'] = row['close']
                    
                    # 删除临时列
                    df_merged = df_merged.drop(columns=['_orig_idx'], errors='ignore')
                
                df_signals = df_merged.reset_index(drop=True)
                
                if verbose and df_signals['close'].isna().any():
                    nan_count = df_signals['close'].isna().sum()
                    total_count = len(df_signals)
                    print(f"警告: 合并后有 {nan_count}/{total_count} 行没有匹配到价格数据，这些信号将被忽略")
            else:
                raise ValueError("df_signals 必须包含 'close' 列，或提供包含 'close' 的 df_trades")
        
        if verbose:
            print("信号数据预览:")
            print(f"\n数据形状: {df_signals.shape}")
            print(f"品种列表: {df_signals['symbol'].unique()}")
        
        # 定义默认合约配置函数（如果未提供）
        if contract_config_func is None:
            def get_contract_config(symbol: str) -> ContractConfig:
                """根据品种代码返回合约配置"""
                # 这里可以根据不同品种设置不同的合约参数
                # 示例：根据品种前缀判断
                if symbol.startswith('A'):
                    # 豆粕等农产品
                    return ContractConfig(
                        symbol=symbol,
                        multiplier=10.0,
                        tick_size=1.0,
                        margin_ratio=0.12,
                        commission_rate=0.00023,
                        commission_per_contract=commission_per_contract,
                        close_today_commission_rate=close_today_commission_rate,
                        close_today_commission_per_contract=close_today_commission_per_contract,
                        max_position=max_position,
                        trade_unit=trade_unit
                    )
                elif symbol.startswith('RB'):
                    # 螺纹钢
                    return ContractConfig(
                        symbol=symbol,
                        multiplier=10.0,
                        tick_size=1.0,
                        margin_ratio=0.12,
                        commission_rate=0.00023,
                        commission_per_contract=commission_per_contract,
                        close_today_commission_rate=close_today_commission_rate,
                        close_today_commission_per_contract=close_today_commission_per_contract,
                        max_position=max_position,
                        trade_unit=trade_unit
                    )
                else:
                    # 默认配置
                    return ContractConfig(
                        symbol=symbol,
                        multiplier=10.0,
                        tick_size=1.0,
                        margin_ratio=0.12,
                        commission_rate=0.00023,
                        commission_per_contract=commission_per_contract,
                        close_today_commission_rate=close_today_commission_rate,
                        close_today_commission_per_contract=close_today_commission_per_contract,
                        max_position=max_position,
                        trade_unit=trade_unit
                    )
            contract_config_func = get_contract_config
        
        # 创建多品种回测
        multi_backtest = MultiSymbolBacktest(
            df_signals=df_signals,
            contract_config_func=contract_config_func,
            initial_capital=initial_capital,
            slippage=slippage,
            quantity=quantity,
            stop_loss_points=stop_loss_points,
            stop_loss_percent=stop_loss_percent,
            take_profit_points=take_profit_points,
            take_profit_percent=take_profit_percent,
            trailing_stop_pct=trailing_stop_pct,
            max_drawdown=max_drawdown,
            margin_call_level=margin_call_level
        )
        
        return multi_backtest


# ==================== 汇总结果导出和可视化函数 ====================

def export_summary_metrics(
    summary_metrics: Dict,
    symbol_metrics: Dict[str, Dict]
) -> pd.DataFrame:
    """
    导出汇总指标为DataFrame（中文列名）
    
    参数:
    summary_metrics: 汇总指标字典
    symbol_metrics: 各品种指标字典
    
    返回:
    summary_df: 汇总指标DataFrame（中文列名）
    """
    # 汇总指标映射（英文 -> 中文）
    summary_mapping = {
        'total_initial_capital': '初始资金',
        'total_final_equity': '最终权益',
        'total_pnl': '总盈亏',
        'total_return': '总收益率',
        'annual_return': '年化收益率',
        'max_drawdown': '最大回撤',
        'calmar_ratio': '卡玛比率',
        'sharpe_ratio': '夏普比率',
        'trade_count': '交易次数',
        'winning_trades': '盈利次数',
        'losing_trades': '亏损次数',
        'win_rate': '胜率',
        'profit_factor': '盈亏比',
        'avg_pnl_per_trade': '平均每笔盈亏',
        'total_commission': '总手续费',
        'num_symbols': '品种数量',
        'backtest_days': '回测天数'
    }
    
    # 创建汇总指标DataFrame
    summary_data = []
    
    # 添加汇总行
    summary_row = {'指标类型': '汇总'}
    for key, value in summary_metrics.items():
        chinese_name = summary_mapping.get(key, key)
        if value is None:
            summary_row[chinese_name] = '-'
        elif isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                summary_row[chinese_name] = '-'
            elif 'return' in key or 'rate' in key or 'ratio' in key or 'drawdown' in key:
                summary_row[chinese_name] = f"{value*100:.2f}%" if 'ratio' not in key else f"{value:.4f}"
            elif 'capital' in key or 'equity' in key or 'pnl' in key or 'commission' in key:
                summary_row[chinese_name] = f"{value:.2f}"
            else:
                summary_row[chinese_name] = f"{value:.4f}"
        else:
            summary_row[chinese_name] = value
    summary_data.append(summary_row)
    
    # 添加各品种指标
    symbol_mapping = {
        'initial_capital': '初始资金',
        'final_equity': '最终权益',
        'total_pnl': '总盈亏',
        'total_return': '总收益率',
        'annual_return': '年化收益率',
        'max_drawdown': '最大回撤',
        'calmar_ratio': '卡玛比率',
        'sharpe_ratio': '夏普比率',
        'sortino_ratio': '索提诺比率',
        'trade_count': '交易次数',
        'winning_trades': '盈利次数',
        'losing_trades': '亏损次数',
        'win_rate': '胜率',
        'profit_factor': '盈亏比',
        'avg_pnl_per_trade': '平均每笔盈亏',
        'avg_win': '平均盈利',
        'avg_loss': '平均亏损',
        'total_commission': '总手续费'
    }
    
    for symbol, metrics in symbol_metrics.items():
        symbol_row = {'指标类型': symbol}
        for key, value in metrics.items():
            chinese_name = symbol_mapping.get(key, key)
            if value is None:
                symbol_row[chinese_name] = '-'
            elif isinstance(value, float):
                if np.isnan(value) or np.isinf(value):
                    symbol_row[chinese_name] = '-'
                elif 'return' in key or 'rate' in key or 'ratio' in key or 'drawdown' in key:
                    symbol_row[chinese_name] = f"{value*100:.2f}%" if 'ratio' not in key else f"{value:.4f}"
                elif 'capital' in key or 'equity' in key or 'pnl' in key or 'commission' in key or 'win' in key or 'loss' in key:
                    symbol_row[chinese_name] = f"{value:.2f}"
                else:
                    symbol_row[chinese_name] = f"{value:.4f}"
            else:
                symbol_row[chinese_name] = value
        summary_data.append(symbol_row)
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


def plot_trade_returns(
    trades_df: pd.DataFrame,
    initial_capital: Optional[float] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
):
    """
    可视化交易记录收益率（精致版）
    
    参数:
    trades_df: 交易记录DataFrame，必须包含 'pnl', 'pnl_pct', 'exit_time' 列
    initial_capital: 初始资金（用于计算正确的累积收益率，如果为None则使用近似方法）
    save_path: 保存路径（可选）
    figsize: 图表大小
    """
    if not HAS_MATPLOTLIB:
        print("警告: matplotlib未安装，无法生成图表")
        return
    
    if trades_df.empty:
        print("警告: 交易记录为空，无法生成图表")
        return
    
    # 确保有必要的列
    if 'pnl_pct' not in trades_df.columns and 'pnl' in trades_df.columns:
        # 如果没有pnl_pct，尝试计算
        if 'entry_price' in trades_df.columns and 'exit_price' in trades_df.columns:
            trades_df = trades_df.copy()
            trades_df['pnl_pct'] = (trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price']
            # 如果是空仓，需要调整符号
            if 'side' in trades_df.columns:
                trades_df.loc[trades_df['side'] == 'SHORT', 'pnl_pct'] *= -1
        else:
            print("警告: 缺少必要的列来计算收益率")
            return
    
    # 按时间排序（如果是多品种，按时间统一排序）
    if 'exit_time' in trades_df.columns:
        trades_df = trades_df.sort_values('exit_time').copy()
        x_data = trades_df['exit_time']
        x_label = '平仓时间'
    else:
        trades_df = trades_df.reset_index(drop=True)
        x_data = range(len(trades_df))
        x_label = '交易序号'
    
    # ========== 修复：正确计算累积收益率 ==========
    # 基于累计盈亏金额（而不是累加百分比）来计算累积收益率
    # 这样与 plot_equity_curve 的计算逻辑一致
    if initial_capital is not None and initial_capital > 0:
        # 如果有初始资金，使用正确的方法：累积收益率 = 累计盈亏 / 初始资金
        cumulative_pnl = trades_df['pnl'].cumsum()
        trades_df['cumulative_return'] = cumulative_pnl / initial_capital
        # 总收益率也基于累计盈亏计算
        total_return_pnl = cumulative_pnl.iloc[-1] if len(cumulative_pnl) > 0 else 0
        total_return = (total_return_pnl / initial_capital) * 100
    else:
        # 如果没有提供初始资金，尝试从交易数据估算
        if 'pnl' in trades_df.columns:
            cumulative_pnl = trades_df['pnl'].cumsum()
            # 估算初始资金：假设初始资金约为最大累计亏损绝对值的某个倍数
            # 或者使用第一笔交易的开仓金额的某个倍数
            if len(cumulative_pnl) > 0:
                min_cumulative_pnl = cumulative_pnl.min()
                max_cumulative_pnl = cumulative_pnl.max()
                
                # 方法1：基于累计盈亏的范围估算（假设初始资金至少是最大亏损的2倍）
                if min_cumulative_pnl < 0:
                    estimated_capital = abs(min_cumulative_pnl) * 2
                elif 'entry_price' in trades_df.columns and 'quantity' in trades_df.columns:
                    # 方法2：基于第一笔交易的开仓金额估算（假设初始资金是开仓金额的10倍）
                    first_trade_value = trades_df.iloc[0]['entry_price'] * trades_df.iloc[0]['quantity'] * 10
                    estimated_capital = max(first_trade_value, 100000)  # 至少10万
                else:
                    # 方法3：如果无法估算，使用默认值
                    estimated_capital = 100000
                
                if estimated_capital > 0:
                    trades_df['cumulative_return'] = cumulative_pnl / estimated_capital
                    total_return_pnl = cumulative_pnl.iloc[-1]
                    total_return = (total_return_pnl / estimated_capital) * 100
                else:
                    # 如果估算失败，使用简单的pnl_pct累加（不准确，但至少有个趋势）
                    trades_df['cumulative_return'] = trades_df['pnl_pct'].cumsum()
                    total_return = trades_df['cumulative_return'].iloc[-1] * 100 if len(trades_df) > 0 else 0
            else:
                trades_df['cumulative_return'] = trades_df['pnl_pct'].cumsum()
                total_return = trades_df['cumulative_return'].iloc[-1] * 100 if len(trades_df) > 0 else 0
        else:
            # 如果没有pnl列，只能使用pnl_pct累加（不准确）
            trades_df['cumulative_return'] = trades_df['pnl_pct'].cumsum()
            total_return = trades_df['cumulative_return'].iloc[-1] * 100 if len(trades_df) > 0 else 0
    # ============================================
    
    # 计算统计指标
    win_trades = trades_df[trades_df['pnl_pct'] > 0]
    lose_trades = trades_df[trades_df['pnl_pct'] <= 0]
    win_rate = len(win_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    avg_return = trades_df['pnl_pct'].mean() * 100
    avg_win = win_trades['pnl_pct'].mean() * 100 if len(win_trades) > 0 else 0
    avg_loss = lose_trades['pnl_pct'].mean() * 100 if len(lose_trades) > 0 else 0
    max_return = trades_df['pnl_pct'].max() * 100
    min_return = trades_df['pnl_pct'].min() * 100
    
    # 设置专业配色方案
    try:
        if 'seaborn-v0_8-darkgrid' in plt.style.available:
            plt.style.use('seaborn-v0_8-darkgrid')
        elif 'seaborn-darkgrid' in plt.style.available:
            plt.style.use('seaborn-darkgrid')
        else:
            plt.style.use('default')
    except:
        plt.style.use('default')
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, 
                          left=0.08, right=0.95, top=0.95, bottom=0.08)
    
    # 定义颜色
    color_win = '#2ecc71'  # 绿色
    color_loss = '#e74c3c'  # 红色
    color_line = '#3498db'  # 蓝色
    color_bg = '#ecf0f1'  # 浅灰背景
    
    # 子图1: 每笔交易收益率（柱状图）
    ax1 = fig.add_subplot(gs[0, :])
    colors = [color_win if x > 0 else color_loss for x in trades_df['pnl_pct']]
    bars = ax1.bar(x_data, trades_df['pnl_pct'] * 100, color=colors, alpha=0.7, 
                   width=0.8, edgecolor='white', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.2, zorder=0)
    ax1.axhline(y=avg_return, color=color_line, linestyle='--', linewidth=2, 
                label=f'平均收益率: {avg_return:.2f}%', zorder=1)
    font_prop = get_chinese_font_prop()
    ax1.set_ylabel('单笔收益率 (%)', fontsize=13, fontweight='bold', fontproperties=font_prop)
    ax1.set_title('交易记录收益率分布', fontsize=16, fontweight='bold', pad=15, fontproperties=font_prop)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9, shadow=True, prop=font_prop)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#bdc3c7')
    ax1.spines['bottom'].set_color('#bdc3c7')
    
    # 子图2: 累积收益率曲线
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(x_data, trades_df['cumulative_return'] * 100, color=color_line, 
            linewidth=2.5, label='累积收益率', marker='o', markersize=3, alpha=0.8)
    ax2.fill_between(x_data, trades_df['cumulative_return'] * 100, 0, 
                     color=color_line, alpha=0.2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.2, zorder=0)
    ax2.set_xlabel(x_label, fontsize=13, fontweight='bold', fontproperties=font_prop)
    ax2.set_ylabel('累积收益率 (%)', fontsize=13, fontweight='bold', fontproperties=font_prop)
    ax2.set_title('累积收益率曲线', fontsize=16, fontweight='bold', pad=15, fontproperties=font_prop)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.legend(loc='best', fontsize=11, framealpha=0.9, shadow=True, prop=font_prop)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#bdc3c7')
    ax2.spines['bottom'].set_color('#bdc3c7')
    
    # 子图3: 收益率分布直方图
    ax3 = fig.add_subplot(gs[2, 0])
    n_bins = min(30, max(10, len(trades_df) // 5))
    ax3.hist(trades_df['pnl_pct'] * 100, bins=n_bins, color=color_line, alpha=0.7, 
            edgecolor='white', linewidth=1.5)
    ax3.axvline(x=avg_return, color='red', linestyle='--', linewidth=2, 
               label=f'平均值: {avg_return:.2f}%')
    ax3.set_xlabel('收益率 (%)', fontsize=12, fontweight='bold', fontproperties=font_prop)
    ax3.set_ylabel('频数', fontsize=12, fontweight='bold', fontproperties=font_prop)
    ax3.set_title('收益率分布直方图', fontsize=14, fontweight='bold', pad=10, fontproperties=font_prop)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.legend(fontsize=10, framealpha=0.9, prop=font_prop)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#bdc3c7')
    ax3.spines['bottom'].set_color('#bdc3c7')
    
    # 子图4: 统计信息面板
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    # 创建统计信息文本
    stats_text = f"""
    ╔═══════════════════════════════════════╗
    ║        交易统计信息                    ║
    ╠═══════════════════════════════════════╣
    ║ 总交易次数: {len(trades_df):>6} 笔              ║
    ║ 盈利次数:   {len(win_trades):>6} 笔              ║
    ║ 亏损次数:   {len(lose_trades):>6} 笔              ║
    ║ 胜率:       {win_rate:>6.2f}%              ║
    ╠═══════════════════════════════════════╣
    ║ 总收益率:   {total_return:>6.2f}%              ║
    ║ 平均收益率: {avg_return:>6.2f}%              ║
    ║ 平均盈利:   {avg_win:>6.2f}%              ║
    ║ 平均亏损:   {avg_loss:>6.2f}%              ║
    ╠═══════════════════════════════════════╣
    ║ 最大单笔收益: {max_return:>6.2f}%              ║
    ║ 最大单笔亏损: {min_return:>6.2f}%              ║
    ╚═══════════════════════════════════════╝
    """
    
    ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            fontproperties=font_prop,
            family='monospace', bbox=dict(boxstyle='round,pad=1', 
            facecolor=color_bg, edgecolor='#34495e', linewidth=2, alpha=0.9))
    
    # 格式化x轴（如果是时间）
    if 'exit_time' in trades_df.columns:
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            interval = max(1, len(trades_df) // 15)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9, fontproperties=font_prop)
    
    # 添加整体标题
    fig.suptitle('交易收益率分析报告', fontsize=18, fontweight='bold', y=0.98, fontproperties=font_prop)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"交易收益率图表已保存到: {save_path}")
    
    plt.show()


def plot_equity_curve(
    equity_df: pd.DataFrame,
    initial_capital: float,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12)
):
    """
    可视化资金曲线（精致版）
    
    参数:
    equity_df: 权益曲线DataFrame，必须包含 'timestamp' 和 'equity' 列，或索引为时间
    initial_capital: 初始资金
    save_path: 保存路径（可选）
    figsize: 图表大小
    """
    if not HAS_MATPLOTLIB:
        print("警告: matplotlib未安装，无法生成图表")
        return
    
    if equity_df.empty:
        print("警告: 权益曲线数据为空，无法生成图表")
        return
    
    # 处理数据
    equity_df = equity_df.copy()
    
    # 如果timestamp是索引，重置为列
    if equity_df.index.name == 'timestamp' or (isinstance(equity_df.index, pd.DatetimeIndex) and 'timestamp' not in equity_df.columns):
        equity_df = equity_df.reset_index()
        if 'index' in equity_df.columns:
            equity_df.rename(columns={'index': 'timestamp'}, inplace=True)
    
    # 确保timestamp列存在
    if 'timestamp' not in equity_df.columns:
        print("警告: 无法找到timestamp列")
        return
    
    # 转换timestamp为datetime类型
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    # 如果有多品种，按时间汇总所有品种的权益
    if 'symbol' in equity_df.columns:
        # 多品种，按时间汇总
        equity_series = equity_df.groupby('timestamp')['equity'].sum().sort_index()
    elif 'equity' in equity_df.columns:
        # 单品种
        equity_df = equity_df.set_index('timestamp').sort_index()
        equity_series = equity_df['equity']
    else:
        # 假设第一列是权益
        equity_df = equity_df.set_index('timestamp').sort_index()
        equity_series = equity_df.iloc[:, 0]
    
    # 计算收益率和回撤
    equity_array = equity_series.values
    returns = np.diff(equity_array) / equity_array[:-1]
    returns = np.concatenate([[0], returns])  # 第一天的收益率为0
    
    # 计算回撤
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max * 100
    
    # 计算统计指标
    final_equity = equity_array[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100
    max_dd = abs(drawdown.min())
    max_dd_idx = equity_series.index[np.argmin(drawdown)]
    
    # 计算年化收益率
    if len(equity_series) > 1:
        days = (equity_series.index[-1] - equity_series.index[0]).days
        if days > 0:
            annual_return = ((final_equity / initial_capital) ** (365.0 / days) - 1) * 100
        else:
            annual_return = 0.0
    else:
        annual_return = 0.0
    
    # 计算夏普比率（简化版）
    if len(returns) > 1:
        returns_clean = returns[~np.isnan(returns)]
        if len(returns_clean) > 0 and returns_clean.std() > 0:
            sharpe_ratio = returns_clean.mean() / returns_clean.std() * np.sqrt(252)  # 假设252个交易日
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0
    
    # 计算卡玛比率
    calmar_ratio = annual_return / max_dd if max_dd > 0 else 0.0
    
    # 计算波动率
    if len(returns) > 1:
        returns_clean = returns[~np.isnan(returns)]
        volatility = returns_clean.std() * np.sqrt(252) * 100 if len(returns_clean) > 0 else 0.0
    else:
        volatility = 0.0
    
    # 设置专业配色方案
    try:
        if 'seaborn-v0_8-darkgrid' in plt.style.available:
            plt.style.use('seaborn-v0_8-darkgrid')
        elif 'seaborn-darkgrid' in plt.style.available:
            plt.style.use('seaborn-darkgrid')
        else:
            plt.style.use('default')
    except:
        plt.style.use('default')
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, 
                          left=0.08, right=0.95, top=0.95, bottom=0.08)
    
    # 定义颜色
    color_equity = '#3498db'  # 蓝色
    color_return_pos = '#2ecc71'  # 绿色
    color_return_neg = '#e74c3c'  # 红色
    color_drawdown = '#e74c3c'  # 红色
    color_bg = '#ecf0f1'  # 浅灰背景
    color_initial = '#95a5a6'  # 灰色
    
    # 子图1: 权益曲线（主图，占两列）
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity_series.index, equity_series.values, color=color_equity, 
            linewidth=3, label='账户权益', zorder=3, alpha=0.9)
    ax1.fill_between(equity_series.index, equity_series.values, initial_capital, 
                    where=(equity_series.values >= initial_capital), 
                    color=color_return_pos, alpha=0.2, label='盈利区域')
    ax1.fill_between(equity_series.index, equity_series.values, initial_capital, 
                    where=(equity_series.values < initial_capital), 
                    color=color_return_neg, alpha=0.2, label='亏损区域')
    ax1.axhline(y=initial_capital, color=color_initial, linestyle='--', 
               linewidth=2, label=f'初始资金: {initial_capital:,.0f}元', zorder=2)
    ax1.axhline(y=final_equity, color=color_equity, linestyle=':', 
               linewidth=1.5, alpha=0.7, zorder=1)
    
    # 标注最高点和最低点
    max_equity_idx = equity_series.idxmax()
    min_equity_idx = equity_series.idxmin()
    ax1.plot(max_equity_idx, equity_series.max(), 'o', color='gold', 
            markersize=10, zorder=4, label='最高权益')
    ax1.plot(min_equity_idx, equity_series.min(), 'o', color='darkred', 
            markersize=10, zorder=4, label='最低权益')
    
    font_prop = get_chinese_font_prop()
    ax1.set_ylabel('账户权益 (元)', fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax1.set_title('资金曲线', fontsize=18, fontweight='bold', pad=20, fontproperties=font_prop)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9, shadow=True, ncol=3, prop=font_prop)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#bdc3c7')
    ax1.spines['bottom'].set_color('#bdc3c7')
    
    # 添加统计信息文本框
    stats_text = f'最终权益: {final_equity:,.2f}元  |  总收益率: {total_return:.2f}%  |  年化收益率: {annual_return:.2f}%'
    ax1.text(0.5, 0.02, stats_text, transform=ax1.transAxes,
            fontsize=12, verticalalignment='bottom', horizontalalignment='center',
            fontproperties=font_prop,
            bbox=dict(boxstyle='round,pad=0.8', facecolor=color_bg, 
            edgecolor='#34495e', linewidth=1.5, alpha=0.9))
    
    # 子图2: 每日收益率
    ax2 = fig.add_subplot(gs[1, :])
    colors = [color_return_pos if x > 0 else color_return_neg for x in returns]
    bars = ax2.bar(equity_series.index, returns * 100, color=colors, alpha=0.7, 
                  width=0.8, edgecolor='white', linewidth=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.2, zorder=0)
    ax2.axhline(y=returns.mean() * 100, color=color_equity, linestyle='--', 
               linewidth=2, label=f'平均收益率: {returns.mean()*100:.2f}%', zorder=1)
    ax2.set_ylabel('日收益率 (%)', fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax2.set_title('每日收益率分布', fontsize=18, fontweight='bold', pad=20, fontproperties=font_prop)
    ax2.legend(loc='best', fontsize=11, framealpha=0.9, shadow=True, prop=font_prop)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#bdc3c7')
    ax2.spines['bottom'].set_color('#bdc3c7')
    
    # 子图3: 回撤曲线
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.fill_between(equity_series.index, drawdown, 0, color=color_drawdown, 
                    alpha=0.4, label='回撤区域', zorder=1)
    ax3.plot(equity_series.index, drawdown, color=color_drawdown, 
            linewidth=2.5, label='回撤曲线', zorder=2)
    
    # 标注最大回撤点
    ax3.plot(max_dd_idx, drawdown.min(), 'o', color='darkred', 
            markersize=12, zorder=3, label=f'最大回撤: {max_dd:.2f}%')
    
    ax3.set_xlabel('时间', fontsize=13, fontweight='bold', fontproperties=font_prop)
    ax3.set_ylabel('回撤 (%)', fontsize=13, fontweight='bold', fontproperties=font_prop)
    ax3.set_title('回撤曲线', fontsize=16, fontweight='bold', pad=15, fontproperties=font_prop)
    ax3.legend(loc='lower right', fontsize=11, framealpha=0.9, shadow=True, prop=font_prop)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#bdc3c7')
    ax3.spines['bottom'].set_color('#bdc3c7')
    ax3.set_ylim(bottom=min(drawdown.min() * 1.1, -1))
    
    # 子图4: 统计信息面板
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    # 创建详细的统计信息文本
    stats_text = f"""
    ╔═══════════════════════════════════════╗
    ║        资金曲线统计信息                ║
    ╠═══════════════════════════════════════╣
    ║ 初始资金:   {initial_capital:>12,.2f} 元    ║
    ║ 最终权益:   {final_equity:>12,.2f} 元    ║
    ║ 总收益:     {final_equity-initial_capital:>12,.2f} 元    ║
    ╠═══════════════════════════════════════╣
    ║ 总收益率:   {total_return:>12.2f}%        ║
    ║ 年化收益率: {annual_return:>12.2f}%        ║
    ║ 最大回撤:   {max_dd:>12.2f}%        ║
    ╠═══════════════════════════════════════╣
    ║ 夏普比率:   {sharpe_ratio:>12.2f}          ║
    ║ 卡玛比率:   {calmar_ratio:>12.2f}          ║
    ║ 波动率:     {volatility:>12.2f}%        ║
    ╠═══════════════════════════════════════╣
    ║ 最高权益:   {equity_series.max():>12,.2f} 元    ║
    ║ 最低权益:   {equity_series.min():>12,.2f} 元    ║
    ║ 交易天数:   {len(equity_series):>12} 天        ║
    ╚═══════════════════════════════════════╝
    """
    
    ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            fontproperties=font_prop,
            family='monospace', bbox=dict(boxstyle='round,pad=1', 
            facecolor=color_bg, edgecolor='#34495e', linewidth=2, alpha=0.9))
    
    # 格式化x轴
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        if len(equity_series) > 30:
            interval = max(1, len(equity_series) // 20)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9, fontproperties=font_prop)
    
    # 添加整体标题
    fig.suptitle('资金曲线分析报告', fontsize=20, fontweight='bold', y=0.98, fontproperties=font_prop)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"资金曲线图表已保存到: {save_path}")
    
    plt.show()


def export_trades(trades_df: pd.DataFrame, 
                  file_path: Optional[str] = None,
                  format: str = 'csv') -> pd.DataFrame:
    """
    导出交易记录为DataFrame并保存到文件
    
    参数:
    trades_df: 交易记录DataFrame
    file_path: 保存路径（如果为None，则使用默认路径）
    format: 保存格式，'csv' 或 'excel'
    
    返回:
    格式化后的交易记录DataFrame
    """
    if trades_df is None or len(trades_df) == 0:
        print("警告: 没有交易记录可导出")
        return pd.DataFrame()
    
    # 确保所有必需的列都存在
    required_columns = [
        'entry_time', 'exit_time', 'side', 'entry_price', 'exit_price',
        'quantity', 'pnl', 'pnl_pct', 'commission', 'exit_reason', 'symbol'
    ]
    
    # 检查并添加缺失的列
    for col in required_columns:
        if col not in trades_df.columns:
            if col == 'symbol':
                # 如果没有symbol列，尝试从其他信息推断，否则使用默认值
                trades_df[col] = 'UNKNOWN'  # 如果没有symbol列，添加默认值
            else:
                trades_df[col] = None  # 其他列填充None
    
    # 确保symbol列不为空（如果是单品种回测，可能没有symbol列）
    if 'symbol' in trades_df.columns and trades_df['symbol'].isna().any():
        trades_df['symbol'] = trades_df['symbol'].fillna('UNKNOWN')
    
    # 按指定顺序重新排列列
    export_df = trades_df[required_columns].copy()
    
    # 统一格式化时间戳为 YYYY-MM-DD HH:MM:SS
    if 'entry_time' in export_df.columns:
        export_df['entry_time'] = pd.to_datetime(export_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    if 'exit_time' in export_df.columns:
        export_df['exit_time'] = pd.to_datetime(export_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 格式化数值列（保持原始格式，只做精度调整）
    if 'entry_price' in export_df.columns:
        export_df['entry_price'] = export_df['entry_price'].round(2)
    if 'exit_price' in export_df.columns:
        export_df['exit_price'] = export_df['exit_price'].round(2)
    if 'pnl' in export_df.columns:
        export_df['pnl'] = export_df['pnl'].round(2)
    if 'pnl_pct' in export_df.columns:
        export_df['pnl_pct'] = export_df['pnl_pct'].round(6)  # 保持小数格式（0.02 表示 2%）
    if 'commission' in export_df.columns:
        export_df['commission'] = export_df['commission'].round(2)
    
    # 保存到文件
    if file_path is None:
        if format == 'csv':
            file_path = 'trades_record.csv'
        else:
            file_path = 'trades_record.xlsx'
    
    try:
        if format.lower() == 'csv':
            export_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"交易记录已保存到: {file_path}")
        elif format.lower() == 'excel':
            export_df.to_excel(file_path, index=False, engine='openpyxl')
            print(f"交易记录已保存到: {file_path}")
        else:
            print(f"警告: 不支持的格式 '{format}'，仅保存为CSV")
            export_df.to_csv(file_path.replace('.xlsx', '.csv'), index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")
        print("但DataFrame已准备好，可以手动保存")
    
    return export_df


def get_trade_price_data(df_signals):
    """
    根据df_signals生成交易价格数据，通过接口获取分钟线数据
    
    参数:
    df_signals: DataFrame，必须包含 'symbol' 和日期列（'date' 或 'data'）
    
    返回:
    df_trades: DataFrame，包含 'symbol', 'date', 'close' 等列
    """
    # 检查必需的列
    if 'symbol' not in df_signals.columns:
        raise ValueError("df_signals 缺少必需的列: 'symbol'")
    
    # 查找日期列（可能是 'date' 或 'data'）
    date_col = None
    if 'date' in df_signals.columns:
        date_col = 'date'
    elif 'data' in df_signals.columns:
        date_col = 'data'
    else:
        raise ValueError("df_signals 缺少日期列，请确保包含 'date' 或 'data' 列")
    
    # 确保 date 列是 datetime 类型
    df = df_signals.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # 获取日期范围
    start_date = df[date_col].min().strftime('%Y%m%d')
    end_date = df[date_col].max().strftime('%Y%m%d')
    
    # 获取所有唯一的 symbols
    symbols = df['symbol'].unique().tolist()
    
    # 调用接口获取数据
    df_trades = panda_data.get_market_min_data(
        symbol=symbols,
        start_date=start_date,
        end_date=end_date,
        symbol_type="future"
    )
    
    return df_trades


if __name__ == '__main__':
    # 示例：使用 df_signals 进行多品种回测
    # df_signals 应包含列: 'symbol', 'date', 'close', 'signal'
    
    # 方式1: 如果已有 df_signals 数据
    # df_signals = pd.DataFrame({
    #     'symbol': ['A2507', 'A2507', 'RB2505', 'RB2505'],
    #     'date': ['2025-01-01', '2025-01-02', '2025-01-01', '2025-01-02'],
    #     'close': [3500, 3510, 3800, 3810],
    #     'signal': [1, 0, -1, 1]
    # })
    import matplotlib.font_manager
    import matplotlib

    # 尝试重建字体缓存（某些版本可能不支持，使用try-except包装）
    try:
        # 检查并尝试不同的重建方法
        if hasattr(matplotlib.font_manager, '_rebuild'):
            matplotlib.font_manager._rebuild()
        elif hasattr(matplotlib.font_manager.fontManager, '_rebuild'):
            matplotlib.font_manager.fontManager._rebuild()
        # 如果方法不存在，跳过重建（不影响程序运行）
    except (AttributeError, Exception) as e:
        # 如果重建失败，不影响程序运行
        pass  # 静默处理，字体设置已在导入时完成
    # 方式2: 从数据生成信号（当前示例）
    # df_data_min = panda_data.get_market_min_data(symbol='A2507', start_date='20250101', end_date='20250131',
    #                                       symbol_type="future")

    df_signals = pd.read_parquet(r'C:\Users\cuiji\Desktop\Hackathon\signal.parquet')
    df_signals = df_signals.sort_values('data', ascending=True)
    df_signals = df_signals.rename(columns={'data':'date'})

    # 生成信号
    # df_signals = generate_ma_signals(df_data_min, ma_short=7, ma_long=23)

    # 实际交易价格表
    # df_trades = df_signals[['symbol', 'date', 'close']].copy()

    df_trades_all= get_trade_price_data(df_signals)
    df_trades = df_trades_all[['symbol', 'date', 'close']].copy()
    df_trades = df_trades.sort_values('date', ascending=True)

    # 假设模型训练之后的 带信号的表
    df_signals = df_signals[['symbol', 'date', 'signal']].copy()


    # ==================== 回测参数配置 ====================
    # 资金与账户参数
    initial_capital = 10000000.0  # 初始本金
    margin_call_level = 0.0  # 强平线（可用保证金<=此值时强平）
    
    # 交易成本参数（最易被忽略但影响最大）
    slippage = 0.1  # 滑点（每手跳数）
    commission_per_contract = 0.0  # 固定手续费（按手，如3元/手、5元/手）
    close_today_commission_rate = 0.0  # 平今仓手续费率（部分品种更高，如0.0003表示3倍普通手续费）
    close_today_commission_per_contract = 0.0  # 平今仓固定手续费（按手）
    
    # 合约与品种参数
    max_position = 5000  # 单品种持仓限额（交易所规定）
    trade_unit = 1  # 最小交易手数
    
    # 交易规则与执行参数
    quantity = 1  # 每次交易手数
    
    # 风控与强平参数
    stop_loss_points = None  # 固定止损（点数，如50点），与stop_loss_percent二选一
    stop_loss_percent = None  # 百分比止损（2%），与stop_loss_points二选一
    take_profit_points = None  # 固定止盈（点数，如100点），与take_profit_percent二选一
    take_profit_percent = None # 百分比止盈（4%），与take_profit_percent二选一
    trailing_stop_pct = None  # 移动止损百分比（如0.01表示1%）
    max_drawdown = None  # 单日最大回撤（5%）
    
    # ==================== 创建回测对象 ====================
    multi_backtest = MultiSymbolBacktest.check_backtest_data(
        df_signals=df_signals,
        df_trades=df_trades,  # 如果 df_signals 已包含 close，可以传 None
        contract_config_func=None,  # 使用默认配置函数（可根据需要自定义）
        # 资金与账户参数
        initial_capital=initial_capital,
        margin_call_level=margin_call_level,
        # 交易成本参数
        slippage=slippage,
        commission_per_contract=commission_per_contract,
        close_today_commission_rate=close_today_commission_rate,
        close_today_commission_per_contract=close_today_commission_per_contract,
        # 合约与品种参数
        max_position=max_position,
        trade_unit=trade_unit,
        # 交易规则参数
        quantity=quantity,
        # 风控参数
        stop_loss_points=stop_loss_points,
        stop_loss_percent=stop_loss_percent,
        take_profit_points=take_profit_points,
        take_profit_percent=take_profit_percent,
        trailing_stop_pct=trailing_stop_pct,
        max_drawdown=max_drawdown,
        verbose=True
    )
    
    # 运行回测
    summary_metrics = multi_backtest.run(verbose=True)
    
    # 获取详细结果
    trades_df, equity_df, metrics, symbol_metrics = multi_backtest.get_results()
    
    # ==================== 汇总结果导出和可视化 ====================
    # 1. 导出汇总指标为DataFrame（中文列名）
    summary_df = export_summary_metrics(metrics, symbol_metrics)
    print("\n" + "=" * 60)
    print("汇总指标（中文）")
    print("=" * 60)
    print(summary_df)
    
    # 2. 可视化交易记录收益率
    plot_trade_returns(trades_df, initial_capital=initial_capital, save_path='trade_returns.png')
    
    # 3. 可视化资金曲线
    plot_equity_curve(equity_df, initial_capital=initial_capital, save_path='equity_curve.png')
    
    # 4. 保存汇总指标到CSV
    summary_df.to_csv('backtest_summary.csv', index=False, encoding='utf-8-sig')
    print("\n汇总指标已保存到: backtest_summary.csv")
    
    # 5. 导出交易记录
    trades_export_df = export_trades(trades_df, file_path='trades_record.csv', format='csv')
    print(f"\n交易记录已导出，共 {len(trades_export_df)} 笔交易")
    print("\n交易记录预览（前5条）:")
    print(trades_export_df.head())
    
    # 打印详细结果
    print("\n" + "=" * 60)
    print("详细结果")
    print("=" * 60)
    print("\n交易记录（前10条）:")
    print(trades_df.head(10))
    print("\n各品种指标:")
    for symbol, symbol_metric in symbol_metrics.items():
        print(f"\n{symbol}:")
        print(f"  最终权益: {symbol_metric.get('final_equity', 0):.2f}")
        print(f"  总收益: {symbol_metric.get('total_return', 0)*100:.2f}%")
        print(f"  交易次数: {symbol_metric.get('trade_count', 0)}")
        print(f"  胜率: {symbol_metric.get('win_rate', 0)*100:.2f}%")