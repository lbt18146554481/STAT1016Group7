import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    LabelEncoder
)
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor, NearestNeighbors
from sklearn.svm import SVR, OneClassSVM
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from scipy.stats import zscore
import lightgbm as lgb
from pyod.models.hbos import HBOS
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.lscp import LSCP
from pyod.models.sod import SOD
from pyod.models.sos import SOS
from pyod.models.lof import LOF
import os
from openai import OpenAI
import time
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import GridSearchCV
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# 固定随机种子
SEED = 42
np.random.seed(SEED)

# 添加自动滚动函数
def auto_scroll_to_top():
    js = '''
        <script>
            function scroll() {
                window.scrollTo(0, 0);
            }
            scroll();
        </script>
    '''
    st.markdown(js, unsafe_allow_html=True)

#------------------------------------------导航组件---------------------------------------------------

def show_navigation():
    st.markdown("---")
    col1, col2, col3 = st.columns([5, 1, 1])
    
    with col2:
        if st.session_state.current_step > 1:
            if st.button("Back", key=f"back_{st.session_state.current_step}"):
                if st.session_state.current_step == 7:
                    if "chat_history" in st.session_state:
                        del st.session_state.chat_history
                    if "displayed_messages" in st.session_state:
                        del st.session_state.displayed_messages
                    if "current_response" in st.session_state:
                        del st.session_state.current_response
                    st.session_state.n0 = 0
                st.session_state.current_step -= 1
                st.session_state.trained = False
                st.session_state.show_prediction = False
                auto_scroll_to_top()  # 添加自动滚动
                st.rerun()
    
    with col3:
        if st.session_state.current_step < 7:
            btn_disabled = False
            if st.session_state.current_step == 1:
                btn_disabled = (st.session_state.processed_data1 is None)
            elif st.session_state.current_step == 2:
                btn_disabled = (st.session_state.processed_data2 is None)
            elif st.session_state.current_step == 5:
                btn_disabled = not st.session_state.trained
            elif st.session_state.current_step == 6:
                btn_disabled = not st.session_state.show_prediction
            
            if st.button("Next", 
                        key=f"next_{st.session_state.current_step}",
                        disabled=btn_disabled):
                st.session_state.current_step += 1
                auto_scroll_to_top()  # 添加自动滚动
                st.rerun()
        else:
            if st.button("Restart", key=f"back_{st.session_state.current_step + 1}"):
                st.session_state.current_step = 0
                auto_scroll_to_top()
                st.rerun()

#------------------------------------------- 初始化会话状态--------------------------------------------------

if 'current_step' not in st.session_state:
    st.session_state.update({
        'current_step': 0,
        'raw_data': None,
        'processed_data1': None,
        'processed_data2': None,
        'model': None,
        'model_type': None,
        'X_test': None,
        'y_test': None,
        'y_pred': None,
        'metrics': {'MAE': 0.0, 'MSE': 0.0, 'R2': 0.0},
        'trained': False,
        'show_prediction': False,
        'analysis_result': None,
        'y_test' : None,
        'y_pred' : None,
        'Response2': None,
        'outliers_removed':None,
        'outliers': None,
        'num_outliers': None,
        'features': None, 
        'target': None,
        'n0': 0  # 添加n0变量初始化
    })
    auto_scroll_to_top()

# 在应用开始时添加
st.markdown(
    """
    <style>
        body {
            scroll-behavior: smooth;
        }
        .main {
            scroll-behavior: smooth;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------欢迎页面---------------------------------------------------

if st.session_state.current_step == 0:
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 style="font-size:70px; background: linear-gradient(45deg, #663399, #9370DB); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Birth Rate Prediction Machine</h1>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <style>
        .team-title {
            font-size: 2.5rem;
            color: #666;
            margin-bottom: 1.5rem;
            margin-top: 2rem;
            font-weight: 500;
        }
        .team-members {
            font-size: 2rem;
            color: #888;
            margin-bottom: 3rem;
        }
        .stButton > button {
            font-size: 24px;
            padding: 15px 30px;
            background: linear-gradient(45deg, #663399, #9370DB);
            color: white;
            border: none;
            border-radius: 10px;
            transition: all 0.3s ease;
            width: 200px;
            margin: 2rem auto;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 51, 153, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="team-title">Group Members:</div>', unsafe_allow_html=True)
    st.markdown('<div class="team-members">Gao Shengyuan · Lu Bitong · Shao shenghe · Xu tianjian · Cai kunhuang</div>', unsafe_allow_html=True)
    
    if st.button("Start!", use_container_width=False):
        st.session_state.current_step = 1
        st.rerun()

#------------------------------------------步骤一 处理缺失值----------------------------------------------
#------------------------------------------------------------------------------------------------------

if st.session_state.current_step == 1:
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 style="font-size:55px; background: linear-gradient(45deg, #663399, #9370DB); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Step1: Handle Missing Values</h1>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("",type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            # 将"missing"值转换为NaN
            processed_df = raw_df.replace(['missing', 'Missing', 'MISSING'], np.nan)
            
            # 显示缺失值统计信息
            total_rows = len(processed_df)
            missing_rows = processed_df.isnull().any(axis=1).sum()
            total_missing = processed_df.isnull().sum().sum()
            
            st.markdown('<h1 style="font-size:40px; color:gray;">(1).Handle Missing Data</h1>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total number of rows", f"{total_rows}")
            col2.metric("Number of rows with blanks", f"{missing_rows}")
            col3.metric("Total number of blanks", f"{total_missing}")
            
            
            col1, col2 = st.columns([3, 2])

            with col1:
                fill_method = st.selectbox("Choose Filling Method",
                    options=["Mean 均值填充", "Median 中位数填充", 
                            "Mode 众数填充", "Given 固定值填充","Delete 删除包含缺失值的行"],
                    index=0
                )
            with col2:
                if fill_method == "Given 固定值填充":
                    fill_value = st.number_input("输入填充值")
                else:
                    fill_value = None

            if st.button("Fill Blanks"):
                with st.spinner("正在处理缺失值..."):
                    # 保存原始数据用于显示
                    original_df = processed_df.copy()
                    original_missing_rows = original_df[original_df.isnull().any(axis=1)]
                    
                    # 记录填充前的缺失值数量
                    original_missing = processed_df.isnull().sum().sum()
                    
                    # 首先进行数据类型转换
                    force_numeric_columns = [
                        'GDP per Capita (Current US dollar)',
                        'Population density (per square kilometer)',
                        'Compulsory education coverage rate',
                        'Higher education coverage rate'
                    ]
                    
                    for col in force_numeric_columns:
                        if col in processed_df.columns:
                            try:
                                processed_df[col] = processed_df[col].astype(str).str.replace(',', '').str.replace('、', '').str.replace(' ', '').str.strip()
                                processed_df[col] = processed_df[col].replace('', np.nan)
                                # 转换为数值类型
                                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                            except Exception as e:
                                st.error(f"列 '{col}' 转换为数值类型时出错：{str(e)}")

                    # 然后进行缺失值填充
                    if fill_method == "Delete 删除包含缺失值的行":
                        original_count = len(processed_df)
                        # 保存要删除的行的数据用于显示
                        rows_to_delete = processed_df[processed_df.isnull().any(axis=1)].copy()
                        # 执行删除操作
                        processed_df = processed_df.dropna().reset_index(drop=True)
                        new_count = len(processed_df)
                        drop_count = original_count - new_count
                        
                        # 显示删除的行和保留的行
                        st.success(f"已删除包含缺失值的{drop_count}行，剩余{new_count}行数据！")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("删除的行（包含缺失值）")
                            st.dataframe(rows_to_delete.style.highlight_null(props='background-color: yellow'), height=300)
                        with col2:
                            st.write("保留的行预览")
                            st.dataframe(processed_df.head(50).style.highlight_null(props='background-color: yellow'), height=300)
                    else:
                        # 对所有列进行填充
                        if fill_method == "Mean 均值填充":
                            for col in processed_df.columns:
                                if processed_df[col].dtype in ['int64', 'float64']:
                                    processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
                                else:
                                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                        elif fill_method == "Median 中位数填充":
                            for col in processed_df.columns:
                                if processed_df[col].dtype in ['int64', 'float64']:
                                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                                else:
                                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                        elif fill_method == "Mode 众数填充":
                            for col in processed_df.columns:
                                processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                        elif fill_method == "Given 固定值填充":
                            processed_df = processed_df.fillna(fill_value)

                    # 计算填充后的缺失值数量
                    remaining_missing = processed_df.isnull().sum().sum()
                    filled_count = original_missing - remaining_missing

                    if fill_method == "Delete 删除包含缺失值的行":
                        st.success(f"已删除包含缺失值的{drop_count}行，剩余{new_count}行数据！") 
                    else:
                        if remaining_missing == 0:
                            st.success("填充成功！")
                        else:
                            st.warning(f"已经使用 '{fill_method}' 填充了{filled_count}个缺失值，但仍有{remaining_missing}个缺失值未能填充。")
                            # 显示哪些列还有缺失值
                            cols_with_missing = processed_df.columns[processed_df.isnull().any()].tolist()
                            st.write("以下列仍包含缺失值：", cols_with_missing)

                    st.session_state.processed_data1 = processed_df

                    # 获取包含缺失值的行的索引
                    missing_rows_idx = original_df.isnull().any(axis=1)
                    
                    # 提取包含缺失值的行（处理前和处理后）
                    missing_rows_before = original_df[missing_rows_idx]
                    missing_rows_after = processed_df.loc[missing_rows_before.index]
                    
                    # 使用两列布局显示结果
                    st.subheader("缺失值填充前后对比")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("包含缺失值的行（填充前）")
                        st.dataframe(missing_rows_before.style.highlight_null(props='background-color: yellow'), height=300)
                        
                    with col2:
                        st.write("相同行填充后的结果")
                        st.dataframe(missing_rows_after.style.highlight_null(props='background-color: yellow'), height=300)
                    
                    # 显示填充的统计信息
                    st.info(f"""
                    - 总共发现 {len(missing_rows_before)} 行数据包含缺失值
                    - 填充方法：{fill_method}
                    - 填充的缺失值总数：{filled_count}
                    """)

                if st.session_state.processed_data1  is not None:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        st.session_state.processed_data1.to_excel(writer, index=False, sheet_name='Filled_Data')
        
                    processed_excel = output.getvalue()

                    st.download_button(
            label="Download fliied Data as Excel",
            data=processed_excel,
            file_name="filled_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}")
            st.stop()

    show_navigation()

#--------------------------------------步骤二 数据编码和标准化----------------------------------------------
#-------------------------------------------------------------------------------------------------------

elif st.session_state.current_step == 2:
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 style="font-size:55px; background: linear-gradient(45deg, #663399, #9370DB); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Step2: Data Preprocessing</h1>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data1
    
    columns_to_drop = ['Region', 'Country']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    encoding_method = st.radio("Choose Encoding method", ["None 不编码","Labelencoder 标签编码", "One-Hot-vector encoder 独热编码"], horizontal=True)
    
    scaling_method = st.radio("Choose Scaling Method", ["None 不缩放", "Standardize 标准化", "Normalize 归一化"], horizontal=True)
    
    if st.button("应用转换"):
        with st.spinner("处理中..."):
            original_df = df.copy()
            processed_df = df.copy()
#--------------------------------------------特征编码-----------------------------------------------
            if encoding_method == "One-Hot 独热编码":
                encoder = OneHotEncoder(sparse=False)
                encoded = encoder.fit_transform(processed_df[categorical_cols])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
                processed_df = pd.concat([processed_df.drop(categorical_cols, axis=1), encoded_df], axis=1)
            else:
                for col in categorical_cols:
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col])
#--------------------------------------------特征放缩-----------------------------------------------
            if scaling_method != "None 不缩放" and numeric_cols:
                scaler = StandardScaler() if scaling_method == "Standardize 标准化" else MinMaxScaler()
                processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
            st.session_state.processed_data2 = processed_df
            st.success("数据处理完成！")
            
            st.subheader("数据处理前后对比")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("处理前的数据预览")
                st.dataframe(original_df.head(50), height=150)
                
            with col2:
                st.write("处理后的数据预览")
                st.dataframe(processed_df.head(50), height=150)
            
            st.rerun()
    
    if st.session_state.processed_data2 is not None:
        st.subheader("Brief View After Preprocessing")
        st.dataframe(st.session_state.processed_data2.head(10), height=150)
#-------------------------------------------允许用户下载------------------------------------------------
        output = BytesIO()
    
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.processed_data2.to_excel(writer, index=False, sheet_name='Preprocessed_Data')
    
        processed_excel = output.getvalue()

        st.download_button(
        label="Download Preprocessed Data as Excel",
        data=processed_excel,
        file_name="preprocessed_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    show_navigation()

#----------------------------------------第三步：异常值检测------------------------------------------------
#-------------------------------------------------------------------------------------------------------

elif st.session_state.current_step == 3:
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 style="font-size:55px; background: linear-gradient(45deg, #663399, #9370DB); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Step3: Handle Outliers</h1>', unsafe_allow_html=True)
    
    if st.session_state.processed_data2 is not None:
       
        st.markdown('<h1 style="font-size:40px; color:gray;">Outlier Detection</h1>', unsafe_allow_html=True)
        
        numeric_cols = st.session_state.processed_data2.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        selected_cols = st.multiselect(
            "选择要检测异常值的列",
            options=numeric_cols,
            default=numeric_cols,
            key="outlier_cols"
        )

        outlier_method = st.selectbox(
            "选择异常检测方法",
            options=[
                "Z-Score 标准分数",
                "LOF 局部离群因子",
                "IForest 孤立森林",
                "KNN K最近邻",
                "HBOS 基于直方图",
                "OCSVM 单类支持向量机",
                "ABOD 角度基异常检测",
                "CBLOF 基于聚类",
                "COF 连接异常因子",
                "LSCP 局部敏感性对比投影",
                "PCA 主成分分析",
                "SOD 基于子空间",
                "SOS 基于统计"
            ]
        )

        if outlier_method == "Z-Score 标准分数":
            threshold = st.slider(
                "Z-Score阈值",
                min_value=1.0,
                max_value=20.0,
                value=3.0,
                step=0.1
            )
        elif outlier_method == "LOF 局部离群因子":
            n_neighbors = st.slider(
                "选择邻居数量",
                min_value=2,
                max_value=20,
                value=5
            )
            lof_threshold = st.slider(
                "LOF阈值（越大越严格）",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.1
            )
        elif outlier_method == "IForest 孤立森林":
            contamination = st.slider(
                "预期异常比例",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01
            )
            iforest_threshold = st.slider(
                "异常分数阈值",
                min_value=-0.5,
                max_value=0.5,
                value=0.0,
                step=0.05
            )
        elif outlier_method == "KNN K最近邻":
            n_neighbors = st.slider(
                "选择邻居数量",
                min_value=2,
                max_value=20,
                value=5
            )
            knn_threshold = st.slider(
                "距离阈值百分位数",
                min_value=80,
                max_value=99,
                value=90,
                step=1
            )
        elif outlier_method == "HBOS 基于直方图":
            n_bins = st.slider(
                "直方图箱数",
                min_value=5,
                max_value=50,
                value=10
            )
            hbos_threshold = st.slider(
                "异常分数阈值（越大越严格）",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5
            )
        elif outlier_method == "OCSVM 单类支持向量机":
            nu = st.slider(
                "异常样本比例",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01
            )
            ocsvm_threshold = st.slider(
                "决策函数阈值",
                min_value=-1.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
        elif outlier_method == "ABOD 角度基异常检测":
            n_neighbors = st.slider(
                "选择邻居数量",
                min_value=2,
                max_value=20,
                value=5
            )
            abod_threshold = st.slider(
                "异常分数阈值（越小越严格）",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1
            )
        elif outlier_method == "CBLOF 基于聚类":
            n_clusters = st.slider(
                "选择聚类数量",
                min_value=2,
                max_value=10,
                value=4
            )
            cblof_threshold = st.slider(
                "CBLOF阈值（越大越严格）",
                min_value=1.0,
                max_value=10.0,
                value=2.0,
                step=0.5
            )
        elif outlier_method == "COF 连接异常因子":
            cof_threshold = st.slider(
                "COF阈值（越大越严格）",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.1
            )
        elif outlier_method == "LSCP 局部敏感性对比投影":
            n_bins = st.slider(
                "直方图箱数",
                min_value=5,
                max_value=50,
                value=10
            )
            n_neighbors = st.slider(
                "选择邻居数量",
                min_value=2,
                max_value=20,
                value=5
            )
            lscp_threshold = st.slider(
                "LSCP阈值（越大越严格）",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.1
            )
        elif outlier_method == "PCA 主成分分析":
            pca_threshold = st.slider(
                "PCA阈值（越大越严格）",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1
            )
        elif outlier_method == "SOD 基于子空间":
            sod_threshold = st.slider(
                "SOD阈值（越大越严格）",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1
            )
        elif outlier_method == "SOS 基于统计":
            sos_threshold = st.slider(
                "SOS阈值（越大越严格）",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1
            )
    
        if st.button("Check Outliers"):
            try:
                with st.spinner("正在检测异常值..."):
                    df = st.session_state.processed_data2.copy()
                    X = df[selected_cols]

                    if outlier_method == "Z-Score 标准分数":
                        z_scores = X.apply(zscore)
                        outliers = (z_scores.abs() > threshold).any(axis=1)
                    
                    elif outlier_method == "LOF 局部离群因子":
                        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
                        lof.fit(X)
                        scores = -lof.score_samples(X)
                        outliers = scores > lof_threshold
                    
                    elif outlier_method == "IForest 孤立森林":
                        iforest = IsolationForest(contamination=contamination, random_state=42)
                        iforest.fit(X)
                        scores = -iforest.score_samples(X) 
                        outliers = scores > iforest_threshold
                    
                    elif outlier_method == "KNN K最近邻":
                        knn = NearestNeighbors(n_neighbors=n_neighbors)
                        knn.fit(X)
                        distances, _ = knn.kneighbors(X)
                        threshold_value = np.percentile(distances.mean(axis=1), knn_threshold)
                        outliers = distances.mean(axis=1) > threshold_value
                    
                    elif outlier_method == "HBOS 基于直方图":
                        hbos = HBOS(n_bins=n_bins)
                        scores = hbos.fit(X).decision_scores_
                        outliers = scores > hbos_threshold
                    
                    elif outlier_method == "OCSVM 单类支持向量机":
                        ocsvm = OneClassSVM(kernel='rbf', nu=nu)
                        scores = ocsvm.fit(X).decision_function(X)
                        outliers = scores < ocsvm_threshold
                    
                    elif outlier_method == "ABOD 角度基异常检测":
                        abod = ABOD(n_neighbors=n_neighbors)
                        scores = abod.fit(X).decision_scores_
                        outliers = scores < abod_threshold
                    
                    elif outlier_method == "CBLOF 基于聚类":
                        cblof = CBLOF(n_clusters=n_clusters)
                        scores = cblof.fit(X).decision_scores_
                        outliers = scores > cblof_threshold
                    
                    elif outlier_method == "COF 连接异常因子":
                        cof = COF()
                        scores = cof.fit(X).decision_scores_
                        outliers = scores > cof_threshold
                    
                    elif outlier_method == "LSCP 局部敏感性对比投影":
                        # 创建基础检测器列表
                        detector_list = [
                            LOF(n_neighbors=n_neighbors),  # LOF检测器
                            HBOS(n_bins=n_bins),          # HBOS检测器
                            ABOD(n_neighbors=n_neighbors)  # ABOD检测器
                        ]
                        
                        lscp = LSCP(detector_list)
                        lscp.fit(X)
                        scores = lscp.decision_scores_
                        outliers = scores > lscp_threshold
                    
                    elif outlier_method == "PCA 主成分分析":
                        pca = PCA()
                        X_pca = pca.fit_transform(X)
                        reconstruction = pca.inverse_transform(X_pca)
                        mse = np.mean(np.power(X - reconstruction, 2), axis=1)
                        outliers = mse > np.percentile(mse, pca_threshold)
                    
                    elif outlier_method == "SOD 基于子空间":
                        sod = SOD()
                        scores = sod.fit(X).decision_scores_
                        outliers = scores > sod_threshold
                    
                    elif outlier_method == "SOS 基于统计":
                        sos = SOS()
                        scores = sos.fit(X).decision_scores_
                        outliers = scores > sos_threshold

                    num_outliers = outliers.sum()
                    st.session_state.outliers = outliers
                    st.session_state.num_outliers = num_outliers

                    st.warning(f"发现 {num_outliers} 个异常值！")
                    st.dataframe(df[outliers].head(20), height=200)

            except Exception as e:
                st.warning('请先fill blanks')
                st.error(f"检测异常值时出错：{str(e)}")
                
        if st.button("Remove Outliers", key="remove_outliers"):
            try:
                df = st.session_state.processed_data2.copy()
                outliers = st.session_state.outliers
                df_cleaned = df[~outliers]
                
                outliers_df = df[outliers]
                
                st.session_state.processed_data2 = df_cleaned
                st.session_state.outliers_removed = True
                
                st.success(f"已删除 {st.session_state.num_outliers} 个异常值，剩余 {len(df_cleaned)} 行数据！")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("删除的异常值数据")
                    st.dataframe(outliers_df.head(50), height=150)
                    
                with col2:
                    st.subheader("处理后的数据预览")
                    st.dataframe(df_cleaned.head(50), height=150)
                
            except Exception as e:
                st.warning('请先check outliers!')
                st.error(f"处理异常值时出错：{str(e)}")

            if st.session_state.processed_data2 is not None:
 #-----------------------------------------允许用户下载------------------------------------------------
             output = BytesIO()
             with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                st.session_state.processed_data2.to_excel(writer, index=False, sheet_name='Filled_Data')
    
             processed_excel = output.getvalue()

             st.download_button(
    label="Download cleaned Data as Excel",
    data=processed_excel,
    file_name="filled_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
        

    
    show_navigation()

#--------------------------------------第四步：训练数据可视化------------------------------------------
#--------------------------------------------------------------------------------------------------

elif st.session_state.current_step == 4:
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 style="font-size:55px; background: linear-gradient(45deg, #663399, #9370DB); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Step4: Training Data Visualization</h1>', unsafe_allow_html=True)

    df = st.session_state.processed_data2
    if df is None:
        st.error("数据未加载，请返回上一步")
        st.stop()

    try:
 #--------------------------------------------单特征分析-----------------------------------------------
        st.markdown('<h1 style="font-size:35px; color:gray;">(1).单特征分析</h1>', unsafe_allow_html=True)
        
        selected_feature = st.selectbox(
            "选择要分析的特征",
            options=df.columns,
            key="single_feature_analysis"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 概率分布图
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=df[selected_feature],
                histnorm='probability density',
                name='数据分布',
                opacity=0.7
            ))
            
            # 添加核密度估计曲线
            kde_x = np.linspace(df[selected_feature].min(), df[selected_feature].max(), 100)
            kde = stats.gaussian_kde(df[selected_feature])
            fig_dist.add_trace(go.Scatter(
                x=kde_x,
                y=kde(kde_x),
                mode='lines',
                name='密度估计',
                line=dict(color='red')
            ))
            
            fig_dist.update_layout(
                title='概率分布图',
                xaxis_title=selected_feature,
                yaxis_title='密度',
                height=400
            )
            st.plotly_chart(fig_dist)
        
        with col2:
            # 箱型图
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=df[selected_feature],
                name=selected_feature,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
            fig_box.update_layout(
                title = '箱线图',
                yaxis_title=selected_feature,
                height=400
            )
            st.plotly_chart(fig_box)

        # 显示基本统计信息
        st.subheader("基本统计信息")
        stats_df = pd.DataFrame({
            '统计量': ['均值', '中位数', '标准差', '最小值', '最大值', '偏度', '峰度'],
            '值': [
                df[selected_feature].mean(),
                df[selected_feature].median(),
                df[selected_feature].std(),
                df[selected_feature].min(),
                df[selected_feature].max(),
                df[selected_feature].skew(),
                df[selected_feature].kurtosis()
            ]
        })
        st.dataframe(stats_df.style.format({'值': '{:.3f}'}))

#--------------------------------------------多特征分析-----------------------------------------------
        st.markdown("---")
        st.markdown('<h1 style="font-size:35px; color:gray;">(2).多特征分析</h1>', unsafe_allow_html=True)
        
        selected_features = st.multiselect(
            "选择要分析的特征（建议选择3-6个特征）",
            options=df.columns,
            default=df.columns[:4] if len(df.columns) > 4 else df.columns
        )
        
        if len(selected_features) > 0:
            X_selected = df[selected_features]
            
            # 1. 特征方差分析
            variance = pd.DataFrame({
                '特征': selected_features,
                '方差': X_selected.var().values
            }).sort_values('方差', ascending=False)
            
            fig_var = go.Figure()
            fig_var.add_trace(go.Bar(
                x=variance['特征'],
                y=variance['方差'],
                marker_color='#1f77b4'
            ))
            fig_var.update_layout(
                title='特征方差分析',
                xaxis_title='特征',
                yaxis_title='方差',
                height=400
            )
            st.plotly_chart(fig_var)

            # 2. 相关性热力图
            corr_matrix = X_selected.corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=selected_features,
                y=selected_features,
                colorscale='Viridis'
            ))
            fig_corr.update_layout(
                title='特征相关性热力图',
                height=600,
                width=800
            )
            st.plotly_chart(fig_corr)

            # 3. PCA 3D可视化
            if len(selected_features) >= 3:
                # 检查并处理缺失值
                X_pca = X_selected.copy()
                if X_pca.isna().any().any():
                    st.warning("数据中存在缺失值，将使用均值进行填充以完成PCA分析")
                    X_pca = X_pca.fillna(X_pca.mean())
                
                pca = PCA(n_components=3)
                X_pca_transformed = pca.fit_transform(X_pca)
                
                # 计算解释方差比
                explained_var_ratio = pca.explained_variance_ratio_
                cumulative_var_ratio = np.cumsum(explained_var_ratio)
                
                # 创建数据框
                df_pca = pd.DataFrame(
                    X_pca_transformed, 
                    columns=['PC1', 'PC2', 'PC3']
                )
                
                # 创建3D散点图
                fig = go.Figure()
                
                fig.add_trace(go.Scatter3d(
                    x=df_pca['PC1'],
                    y=df_pca['PC2'],
                    z=df_pca['PC3'],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=df_pca['PC1'],  
                        colorscale='Viridis',
                        opacity=0.8
                    )
                ))
                
                fig.update_layout(
                    title='PCA 3D可视化',
                    scene=dict(
                        xaxis_title='PC1',
                        yaxis_title='PC2',
                        zaxis_title='PC3'
                    ),
                    width=800,
                    height=600
                )
                
                st.plotly_chart(fig)
                
                # 显示解释方差比
                col1, col2, col3 = st.columns(3)
                col1.metric("PC1解释方差比", f"{explained_var_ratio[0]:.2%}")
                col2.metric("PC2解释方差比", f"{explained_var_ratio[1]:.2%}")
                col3.metric("PC3解释方差比", f"{explained_var_ratio[2]:.2%}")
                
                st.markdown(f"**累积解释方差比：{cumulative_var_ratio[2]:.2%}**")

                # 添加主成分组成信息
                st.subheader("主成分组成分析")
                components_df = pd.DataFrame(
                    pca.components_,
                    columns=selected_features,
                    index=['PC1', 'PC2', 'PC3']
                )
                
                # 显示每个主成分的特征权重
                st.write("各特征在主成分中的权重：")
                st.dataframe(components_df.style.background_gradient(cmap='coolwarm'))
                
            else:
                st.warning("请至少选择3个特征以进行PCA 3D可视化")
        
    except Exception as e:
        st.error(f"可视化错误：{str(e)}")

    show_navigation()

#----------------------------------------第五步：模型训练-----------------------------------------
#----------------------------------------------------------------------------------------------

elif st.session_state.current_step == 5:
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 style="font-size:55px; background: linear-gradient(45deg, #663399, #9370DB); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Step5: Model Training</h1>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data2
    if df is None:
        st.error("数据未加载，请返回上一步")
        st.stop()

#--------------------------------------------变量选择-----------------------------------------------
    st.markdown('<h1 style="font-size:35px; color:gray;">(1).选择特征和目标变量</h1>', unsafe_allow_html=True)
    
    # 获取所有列名
    all_columns = df.columns.tolist()
    
    # 选择目标变量
    target = st.selectbox(
        "选择目标变量",
        options=all_columns,
        key="target_selection"
    )
    
    # 选择特征变量（多选）
    features = st.multiselect(
        "选择特征变量",
        options=[col for col in all_columns if col != target],
        default=[col for col in all_columns if col != target],
        key="feature_selection"
    )
    
    # 保存选择的特征和目标变量到session state
    if features and target:
        st.session_state.features = features
        st.session_state.target = target
       
    
    st.markdown('<h1 style="font-size:35px; color:gray;">(2).模型训练</h1>', unsafe_allow_html=True)

   
    
  #--------------------------------------------模型选择-----------------------------------------------
    model_type = st.selectbox(
        "Choose Model Type",
        options=[
            "LinearRegression 线性回归", 
            "DecisionTree 决策树", 
            "RandomForest 随机森林",
            "KNeighbors KNN回归", 
            "SVM 支持向量机", 
            "BayesianRidge 贝叶斯回归",
            "AdaBoost 自适应提升", 
            "XGBoost 极限梯度提升",
            "LightGBM 轻量梯度提升", 
            "GradientBoosting 梯度提升",
            "Ridge 岭回归",
            "Lasso 套索回归"
        ]
    )
    
    # 模型训练列
    train_size = st.slider("Train-Test Split", 0.5, 0.9, 0.8, step=0.05)

# -------------------------------------------超参数调优-------------------------------------------

    tuning_enabled = st.checkbox("手动调参")
    auto_tuning = st.checkbox("自动调参（网格搜索）") if not tuning_enabled else False
    #允许手动调参
    if tuning_enabled:
        if model_type == "DecisionTree 决策树":
            max_depth = st.slider("Max Depth", 1, 50, 3)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
            model_params = {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split
            }
        elif model_type == "RandomForest 随机森林":
            n_estimators = st.slider("Number of Trees", 10, 200, 50)
            max_depth = st.slider("Max Depth", 1, 20, 5)
            model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }
        elif model_type == "AdaBoost 自适应提升":
            n_estimators = st.slider("Number of Estimators", 10, 200, 50)
            learning_rate = st.slider("Learning Rate", 0.01, 2.0, 1.0, step=0.01)
            model_params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate
            }
        elif model_type == "XGBoost 极限梯度提升":
            n_estimators = st.slider("Number of Estimators", 10, 200, 50)
            max_depth = st.slider("Max Depth", 1, 20, 3)
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, step=0.01)
            model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate
            }
        elif model_type == "KNeighbors KNN回归":
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 3)
            weights = st.selectbox("Weights", options=["uniform", "distance"])
            model_params = {
                'n_neighbors': n_neighbors,
                'weights': weights
            }
        elif model_type == "SVM 支持向量机":
            C = st.slider("C (Regularization)", 0.1, 10.0, 1.0, step=0.1)
            kernel = st.selectbox("Kernel", options=["linear", "poly", "rbf", "sigmoid"])
            model_params = {
                'C': C,
                'kernel': kernel
            }
        elif model_type == "BayesianRidge 贝叶斯回归":
            alpha_1 = st.slider("Alpha 1 (noise precision)", 1e-7, 1e-5, 1e-6, format="%.0e")
            alpha_2 = st.slider("Alpha 2 (weights precision)", 1e-7, 1e-5, 1e-6, format="%.0e")
            lambda_1 = st.slider("Lambda 1 (noise precision)", 1e-7, 1e-5, 1e-6, format="%.0e")
            lambda_2 = st.slider("Lambda 2 (weights precision)", 1e-7, 1e-5, 1e-6, format="%.0e")
            model_params = {
                'alpha_1': alpha_1,
                'alpha_2': alpha_2,
                'lambda_1': lambda_1,
                'lambda_2': lambda_2
            }
        elif model_type == "LightGBM 轻量梯度提升":
            n_estimators = st.slider("Number of Estimators", 10, 200, 50)
            max_depth = st.slider("Max Depth", 1, 20, 3)
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, step=0.01)
            num_leaves = st.slider("Number of Leaves", 2, 50, 31)
            model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'num_leaves': num_leaves
            }
        elif model_type == "GradientBoosting 梯度提升":
            n_estimators = st.slider("Number of Estimators", 10, 200, 50)
            max_depth = st.slider("Max Depth", 1, 20, 3)
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, step=0.01)
            min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
            model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'min_samples_split': min_samples_split
            }
        elif model_type == "Ridge 岭回归":
            alpha = st.slider("Alpha (L2正则化强度)", 0.0001, 10.0, 1.0, step=0.1)
            solver = st.selectbox("求解器", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
            model_params = {
                'alpha': alpha,
                'solver': solver
            }
        elif model_type == "Lasso 套索回归":
            alpha = st.slider("Alpha (L1正则化强度)", 0.0001, 10.0, 1.0, step=0.1)
            max_iter = st.slider("最大迭代次数", 100, 2000, 1000, step=100)
            model_params = {
                'alpha': alpha,
                'max_iter': max_iter
            }
        else:
            st.warning("该模型不适用调参")
            model_params = {}

    elif auto_tuning:
        if model_type == "DecisionTree 决策树":
            param_grid = {
                'max_depth': range(3, 11),
                'min_samples_split': range(2, 6)
            }
        elif model_type == "RandomForest 随机森林":
            param_grid = {
                'n_estimators': range(50, 101, 10),
                'max_depth': range(5, 16)
            }
        elif model_type == "AdaBoost 自适应提升":
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 1.0]
            }
        elif model_type == "XGBoost 极限梯度提升":
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        elif model_type == "KNeighbors KNN回归":
            param_grid = {
                'n_neighbors': range(3, 6),
                'weights': ["uniform", "distance"]
            }
        elif model_type == "SVM 支持向量机":
            param_grid = {
                'C': [0.1, 1.0, 2.0],
                'kernel': ["linear", "poly", "rbf", "sigmoid"]
            }
        elif model_type == "BayesianRidge 贝叶斯回归":
            param_grid = {
                'alpha_1': [1e-7, 1e-6, 1e-5],
                'alpha_2': [1e-7, 1e-6, 1e-5],
                'lambda_1': [1e-7, 1e-6, 1e-5],
                'lambda_2': [1e-7, 1e-6, 1e-5]
            }
        elif model_type == "LightGBM 轻量梯度提升":
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [20, 31, 40]
            }
        elif model_type == "GradientBoosting 梯度提升":
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'min_samples_split': [2, 4, 6]
            }
        elif model_type == "Ridge 岭回归":
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky']
            }
        elif model_type == "Lasso 套索回归":
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 1500, 2000]
            }
        else:
            st.warning("该模型不适用调参")
            param_grid = None

    # 训练模型
    if st.button("Train!", use_container_width=True):
        with st.spinner("训练中..."):
            try:
                X = df[st.session_state.features]
                y = df[st.session_state.target]
                
                # 检查并处理缺失值
                if X.isna().any().any() or y.isna().any():
                    st.warning("数据中存在缺失值，将使用均值进行填充以完成训练")
                    X = X.fillna(X.mean())
                    y = y.fillna(y.mean())
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    train_size=train_size,
                    random_state=SEED
                )
                st.session_state.y_test = y_test

                model_map = {
                    "LinearRegression 线性回归": LinearRegression(),
                    "DecisionTree 决策树": DecisionTreeRegressor(random_state=SEED),
                    "RandomForest 随机森林": RandomForestRegressor(random_state=SEED),
                    "KNeighbors KNN回归": KNeighborsRegressor(),
                    "SVM 支持向量机": SVR(),
                    "BayesianRidge 贝叶斯回归": BayesianRidge(),
                    "AdaBoost 自适应提升": AdaBoostRegressor(random_state=SEED),
                    "XGBoost 极限梯度提升": XGBRegressor(random_state=SEED),
                    "LightGBM 轻量梯度提升": lgb.LGBMRegressor(random_state=SEED),
                    "GradientBoosting 梯度提升": GradientBoostingRegressor(random_state=SEED),
                    "Ridge 岭回归": Ridge(random_state=SEED),
                    "Lasso 套索回归": Lasso(random_state=SEED)
                }
                base_model = model_map[model_type]

                # 根据用户选择决定是否使用调参
                if tuning_enabled:
                    # 使用用户设定的参数
                    if model_type == "BayesianRidge 贝叶斯回归":
                        model = type(base_model)(**model_params)
                    else:
                        model = type(base_model)(**model_params, random_state=SEED if hasattr(base_model, 'random_state') else None)
                    model.fit(X_train, y_train)
                    st.success(f"使用参数: {model_params}")
                elif auto_tuning and param_grid:
                    # 使用网格搜索
                    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='r2', verbose=1)
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    st.success(f"最佳超参数: {grid_search.best_params_}")
                else:
                    # 使用默认参数
                    model = base_model
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                st.session_state.update({
                    'model': model,
                    'model_type': model_type,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'metrics': {
                        'MAE': mean_absolute_error(y_test, y_pred),
                        'MSE': mean_squared_error(y_test, y_pred),
                        'R2': r2_score(y_test, y_pred)
                    },
                    'trained': True
                })
                if y_pred is not None:
                    st.success("模型训练成功！")
                    st.session_state.y_pred = y_pred
                else:
                    st.warning('模型训练失败')
            except Exception as e:
                st.error(f"训练失败：{str(e)}")
    show_navigation()

#----------------------------------------第六步：模型性能可视化-----------------------------------------
#---------------------------------------------------------------------------------------------------

elif st.session_state.current_step == 6:
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 style="font-size:55px; background: linear-gradient(45deg, #663399, #9370DB); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Step6: Model Performance</h1>', unsafe_allow_html=True)
    
    if not st.session_state.get('model'):
        st.error("模型未训练，请返回第五步")
        st.stop()
    
#--------------------------------------------展示性能数据-----------------------------------------------
    st.markdown('<h1 style="font-size:40px; color:gray;">(1).Evaluate Model Performance</h1>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{st.session_state.metrics['MAE']:.2f}")
    col2.metric("MSE", f"{st.session_state.metrics['MSE']:.2f}")
    col3.metric("R²", f"{st.session_state.metrics['R2']:.2f}")
    st.write("")

    #预测数据简要
    st.subheader("Real Values vs Prediction Values") 
    y_pred_sample = st.session_state.y_pred[:5]
    y_test_sample = st.session_state.y_test[:5]
    # 创建对比表格
    comparison_df = pd.DataFrame({
    "Prediction Values": y_pred_sample,
    "Real Values": y_test_sample
    })
    st.dataframe(
    comparison_df.style.format("{:.2f}"),  # 保留两位小数
     height=200
    )
    st.write("")
    
    #--------------------------------------------画图-----------------------------------------------
    st.markdown('<h1 style="font-size:40px; color:gray;">(2).Visualization</h1>', unsafe_allow_html=True)
    try:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        model_type = st.session_state.model_type
        
        if model_type == "LinearRegression 线性回归":
            if len(st.session_state.features) == 1:
                st.subheader("Plot 回归图")
                ax1.scatter(st.session_state.X_test, st.session_state.y_test, color='#1f77b4', label='实际值')
                ax1.plot(st.session_state.X_test, st.session_state.y_pred, color='#ff7f0e', linewidth=2, label='预测值')
                ax1.set_xlabel(st.session_state.features[0])
                ax1.set_ylabel(st.session_state.target)
                ax1.set_title("线性回归拟合曲线")
                st.pyplot(fig1)
                st.markdown("""
                **图表说明**  
                - X轴：选择的特征值  
                - Y轴：目标值的实际值与预测值  
                - 红线：线性回归模型的预测拟合线  
                """)
                st.write("")
            else:
                pca = PCA(n_components=1)
                X_pca = pca.fit_transform(st.session_state.X_test)
                st.subheader("Scatter 散点图")
                ax1.scatter(X_pca, st.session_state.y_test, color='#1f77b4', label='实际值')
                ax1.scatter(X_pca, st.session_state.y_pred, color='#ff7f0e', label='预测值')
                ax1.set_title("散点图")
                ax1.set_xlabel("主成分1 (PCA降维)")
                ax1.set_ylabel(st.session_state.target)
                ax1.set_title(f"{model_type}预测结果", pad=20)
                ax1.legend()
                st.pyplot(fig1)
                st.markdown("""
                **图表说明**  
                - X轴：使用PCA降维后的主成分  
                - Y轴：目标值的实际值与预测值  
                - 蓝点代表实际
                - 黄点代表预测
                """)
                st.write("")
        
        elif model_type in ["DecisionTree 决策树", "RandomForest 随机森林"]:
            importance = pd.DataFrame({
                '特征': st.session_state.features,
                '重要性': st.session_state.model.feature_importances_
            }).sort_values('重要性', ascending=True)
            ax1.barh(importance['特征'], importance['重要性'], color='#2ca02c')
            ax1.set_title("特征重要性分析", pad=20)
            ax1.set_xlabel("重要性得分", labelpad=10)
        
        elif model_type == "KNeighbors KNN回归":
            ax1.scatter(st.session_state.y_test, st.session_state.y_pred, alpha=0.6, color='#9467bd')
            ax1.plot([st.session_state.y_test.min(), st.session_state.y_test.max()], 
                    [st.session_state.y_test.min(), st.session_state.y_test.max()], 'k--')
            ax1.set_xlabel("Real Values", labelpad=10)
            ax1.set_ylabel("Prediction Values", labelpad=10)
            ax1.set_title("Real Values vs Prediction Values", pad=20)
        
        elif model_type == "SVM 支持向量机":
            ax1.scatter(st.session_state.X_test.iloc[:, 0], st.session_state.y_test, color='#17becf', label='实际值')
            ax1.scatter(st.session_state.X_test.iloc[:, 0], st.session_state.y_pred, color='#e377c2', label='预测值')
            ax1.set_xlabel(st.session_state.features[0], labelpad=10)
            ax1.set_ylabel(st.session_state.target, labelpad=10)
            ax1.set_title("SVM回归预测分布", pad=20)
            ax1.legend()
        
        st.subheader("Residual 残差图")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        residuals = st.session_state.y_test - st.session_state.y_pred
        ax2.scatter(st.session_state.y_pred, residuals, alpha=0.4, color='#2ca02c', label='Residuals')
        lowess_curve = lowess(residuals, st.session_state.y_pred, frac=0.3)
        ax2.plot(lowess_curve[:, 0], lowess_curve[:, 1], color='#d62728', lw=1.5, label='Lowess Curve')  
        ax2.axhline(0, color='#7f7f7f', linestyle='--', linewidth=1)
        ax2.set_title("Residual", pad=15)
        ax2.set_xlabel("Predited Values", labelpad=10)
        ax2.set_ylabel("Residual", labelpad=10)
        st.pyplot(fig2)
        st.markdown("""
        **图表说明**  
        - X轴：模型预测值  
        - Y轴：预测残差（实际值 - 预测值）  
        - 红线：残差的趋势线  
        - 灰色虚线： 残差为 0 的基准线，用于对比误差的正负分布
        """)
        
    except Exception as e:
        st.error(f"可视化错误：{str(e)}")
    st.session_state.show_prediction = True
    
    show_navigation()

#----------------------------------------第六步：预测新数据--------------------------------------------
#---------------------------------------------------------------------------------------------------

elif st.session_state.current_step == 6:
    st.markdown('<h1 style="font-size:55px; background: linear-gradient(45deg, #663399, #9370DB); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Step6: Model Prediction</h1>', unsafe_allow_html=True)
    
    if not st.session_state.get('model'):
        st.error("模型未训练，请返回第五步")
        st.stop()
    
    # 创建输入表单
    with st.form("prediction_form"):
        st.subheader("Input Features")
        input_data = {}
        selected_features = st.session_state.features
        
        # 动态生成输入框
        cols = st.columns(2)
        for i, feature in enumerate(selected_features):
            with cols[i % 2]:
                input_data[feature] = st.number_input(
                    f"{feature} ",
                    key=f"input_{feature}"
                )
        left, right = st.columns(2)
        if '生育政策' in st.session_state.features:
           with right:
              st.write('有生育政策输入1，无则输入0')
        # 提交按钮
        submitted = st.form_submit_button("Predict!")
        
        if submitted:
            try:
                # 转换输入数据
                input_df = pd.DataFrame([input_data])
                
                # 进行预测
                y_pred = st.session_state.model.predict(input_df)
                
                # 显示结果
                st.success("预测完成！")
                st.metric(
                    label=f"Predicted {st.session_state.target}:",
                    value=f"{y_pred[0]:.2f}"
                )
                st.session_state.show_prediction = True
            except Exception as e:
                st.error(f"预测失败：{str(e)}")
    show_navigation()
    n0 = 0
    st.session_state.n0 = n0
    

#----------------------------------------第七步：数据智能分析-------------------------------------------
#---------------------------------------------------------------------------------------------------
    
elif st.session_state.current_step == 7:
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 style="font-size:55px; background: linear-gradient(45deg, #663399, #9370DB); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Step7: Data Analysis</h1>', unsafe_allow_html=True)
    
    # 初始化对话历史记录
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "displayed_messages" not in st.session_state:
        st.session_state.displayed_messages = []
    if "current_response" not in st.session_state:
        st.session_state.current_response = None

    # 检查必要状态
    if not st.session_state.get('model'):
        st.error("模型未训练，请返回第五步")
        st.stop()

    # 创建一个容器来显示所有对话内容
    chat_container = st.container()

    # 在对话容器中显示内容
    with chat_container:
        # 显示问候语
        if st.session_state.n0 == 0:
            greeting = 'Hi! I am your stat1016 assistant and I can give you analysis based on the \ndata you just produced.'
            placeholder = st.empty()
            displayed_text = ""
            for char in greeting:
                displayed_text += char
                placeholder.markdown(f"**助手回答：**\n```\n{displayed_text}\n```")
                time.sleep(0.0125)
            st.session_state.n0 = 1
            st.markdown("---")

        # 显示历史对话
        if len(st.session_state.displayed_messages) > 0:
            for msg in st.session_state.displayed_messages[:-1]:
                with st.container():
                    if msg['role'] == 'user':
                        st.markdown(f"**您的问题：**\n```\n{msg['content']}\n```")
                    else:
                        st.markdown(f"**助手回答：**\n```\n{msg['content']}\n```")
                    st.markdown("---")

            # 显示最新的对话（如果存在）
            if len(st.session_state.displayed_messages) > 0:
                latest_msg = st.session_state.displayed_messages[-1]
                with st.container():
                    if latest_msg['role'] == 'user':
                        st.markdown(f"**您的问题：**\n```\n{latest_msg['content']}\n```")
                    else:
                        placeholder = st.empty()
                        displayed_text = ""
                        for char in latest_msg['content']:
                            displayed_text += char
                            placeholder.markdown(f"**助手回答：**\n```\n{displayed_text}\n```")
                            time.sleep(0.02)
                    st.markdown("---")

#--------------------------------------------模型性能分析-----------------------------------------------
    st.markdown("### Step 1: Model Analysis")
    analyze_button = st.button("Analyze Model")

    if analyze_button:
        with st.spinner("thinking..."):
            try:
                # 构建数据摘要
                data_summary = {
                    "数据概况": {
                        "样本数量": len(st.session_state.processed_data2),
                        "特征数量": len(st.session_state.features),
                        "特征变量": st.session_state.features,
                        "目标变量": st.session_state.target
                    },
                    "模型信息": {
                        "模型类型": st.session_state.model_type,
                        "MAE": round(st.session_state.metrics['MAE'], 3),
                        "R2": round(st.session_state.metrics['R2'], 3)
                    },
                    "特征重要性": (
                        dict(zip(
                            st.session_state.features,
                            np.squeeze(
                                st.session_state.model.feature_importances_ if hasattr(st.session_state.model, 'feature_importances_')
                                else st.session_state.model.coef_
                            ).round(3)
                        ))
                        if (hasattr(st.session_state.model, 'feature_importances_') or hasattr(st.session_state.model, 'coef_'))
                        and len(st.session_state.features) == len(np.squeeze(
                            st.session_state.model.feature_importances_ if hasattr(st.session_state.model, 'feature_importances_')
                            else st.session_state.model.coef_
                        ))
                        else "无法计算特征重要性"
                    )
                }

                message = f"""
                你是一个专业的数据科学家，你收集了可能影响一个国家人口增长率的数据，并用这些数据训练了模型，然后给模型输入新的数据来预测人口增长率
                请根据以下分析请求和提供的数据摘要，用中文给出专业分析报告,字数大概300多字：

                分析要求：
                1. 解读数据特征与目标变量({st.session_state.target})的关系
                2. 评估当前模型性能并提出改进建议
                3. 分析可能影响预测结果准确性的潜在因素
                4. 给出可操作的政策建议（如存在政策相关特征）

                数据摘要：
                {data_summary}
                """

                # 储存用户问题
                user_message = {"role": "user", "content": "请分析模型结果"}
                st.session_state.displayed_messages.append(user_message)

                client = OpenAI(
                    api_key = 'de226819-d548-4b72-aa45-470adb3bd551',
                    base_url = "https://ark.cn-beijing.volces.com/api/v3",
                )

                response = client.chat.completions.create(
                    model="ep-20250123135734-mwd8w",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": message},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "https://ark-project.tos-cn-beijing.ivolces.com/images/view.jpeg"
                                    }
                                },
                            ],
                        }
                    ],
                )

                full_response = response.choices[0].message.content
                assistant_message = {"role": "assistant", "content": full_response}
                st.session_state.displayed_messages.append(assistant_message)
                st.rerun()

            except Exception as e:
                st.error(f"分析失败：{str(e)}")

  #--------------------------------------------其他问题-----------------------------------------------
    st.markdown("### Step 2: Additional Questions")
    st.markdown("如果您还有其他问题，请在下方输入：")
    
    message2 = st.text_area(
        "",
        placeholder="在此输入您的问题...",
        key="user_question"
    )
    
    submit_button = st.button("Submit Question")

    if submit_button:
        if message2.strip() == "":
            st.warning("请输入您的问题后再提交！")
        else:
            with st.spinner("thinking..."):
                user_message = {"role": "user", "content": message2}
                st.session_state.displayed_messages.append(user_message)

                client = OpenAI(
                    api_key = 'de226819-d548-4b72-aa45-470adb3bd551',
                    base_url = "https://ark.cn-beijing.volces.com/api/v3",
                )

                response2 = client.chat.completions.create(
                    model="ep-20250123135734-mwd8w",
                    messages=[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.displayed_messages]
                )

                full_response = response2.choices[0].message.content
                assistant_message = {"role": "assistant", "content": full_response}
                st.session_state.displayed_messages.append(assistant_message)
                st.rerun()

    show_navigation()
