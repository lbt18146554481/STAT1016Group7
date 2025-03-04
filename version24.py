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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import os
from openai import OpenAI
import time
from scipy.stats import zscore
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import GridSearchCV
from io import BytesIO

# 初始化会话状态
if 'current_step' not in st.session_state:
    st.session_state.update({
        'current_step': 1,
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
        'num_outliers': None
    })

# 固定随机种子
SEED = 42
np.random.seed(SEED)

# 导航组件
def show_navigation():
    st.markdown("---")
    col1, col2, col3 = st.columns([5, 1, 1])
    
    with col2:
        if st.session_state.current_step > 1:
            if st.button("Back", key=f"back_{st.session_state.current_step}"):
                st.session_state.current_step -= 1
                st.session_state.trained = False
                st.session_state.show_prediction = False
                st.rerun()
    
    with col3:
        if st.session_state.current_step < 6:
            btn_disabled = False
            if st.session_state.current_step == 1:
                btn_disabled = (st.session_state.processed_data1 is None)
            elif st.session_state.current_step == 2:
                btn_disabled = (st.session_state.processed_data2 is None)
            elif st.session_state.current_step == 3:
                btn_disabled = not st.session_state.trained
            elif st.session_state.current_step == 4:
                btn_disabled = not st.session_state.show_prediction
            
            if st.button("Next", 
                        key=f"next_{st.session_state.current_step}",
                        disabled=btn_disabled):
                st.session_state.current_step += 1
                st.rerun()
        else:
            if st.button("Restart", key=f"back_{st.session_state.current_step + 1}"):
                st.session_state.current_step = 1
                st.rerun()
           
# 页面配置
if st.session_state.current_step == 1:
    st.markdown('<h1 style="font-size:60px; color:lightgray;">Birth Rate Prediction Machine</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size:20px; color:lightgray;">Group Mates：Gao Shengyuan/Lu Bitong/Shao shenghe/Xu tianjian/Cai kun huang</h3>', unsafe_allow_html=True)

#---------------------------------- 步骤1：数据上传与缺失处理----------------------------------
#------------------------------------------------------------------------------------------

if st.session_state.current_step == 1:
    st.markdown('<h1 style="font-size:50px; color:gray;">Step1: Data Cleaning</h1>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("",type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            st.session_state.raw_data = raw_df
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}")
            st.stop()
        st.markdown('<h1 style="font-size:40px; color:gray;">(1).Handle Missing Data</h1>', unsafe_allow_html=True)
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
                processed_df = st.session_state.raw_data.copy()
                if fill_method == "Delete 删除包含缺失值的行":
                    original_count = len(processed_df)
                    processed_df = processed_df.dropna()
                    new_count = len(processed_df)
                    drop_count = original_count - new_count
                else:
                    for col in processed_df.columns:
                        if processed_df[col].dtype in ['int64', 'float64']:
                            if fill_method == "Mean 均值填充":
                                processed_df[col].fillna(processed_df[col].mean(), inplace=True)
                            elif fill_method == "Median 中位数填充":
                                processed_df[col].fillna(processed_df[col].median(), inplace=True)
                            elif fill_method == "固定值填充":
                                processed_df[col].fillna(fill_value, inplace=True)
                        else:
                            if fill_method == "Mode 众数填充":
                                processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
                            elif fill_method == "Given 固定值填充":
                                processed_df[col].fillna(fill_value, inplace=True) 
                    st.success("缺失值填充完成！")
                st.session_state.processed_data1 = processed_df

                if fill_method == "Delete 删除包含缺失值的行":
                    st.success(f"已删除包含缺失值的{drop_count}行，剩余{new_count}行数据！")

            if st.session_state.processed_data1  is not None:
                # 创建一个内存缓冲区
                output = BytesIO()
    
                # 使用 Pandas 的 ExcelWriter 和 xlsxwriter 引擎写入 Excel
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.processed_data1.to_excel(writer, index=False, sheet_name='Filled_Data')
    
               # 获取 Excel 文件的二进制内容
                processed_excel = output.getvalue()

              # 添加下载按钮
                st.download_button(
        label="Download fliied Data as Excel",
        data=processed_excel,
        file_name="filled_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
                


        st.markdown('<h1 style="font-size:40px; color:gray;">(2).Handle Outliers</h1>', unsafe_allow_html=True)
        if st.session_state.processed_data1 is not None:
            numeric_cols = st.session_state.processed_data1.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_cols = st.multiselect(
                "Choose columns",
                options=numeric_cols,
                default=numeric_cols
            )
            
            threshold = st.slider(
                "Choose Z-Score Threshold",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.1
            )
        
    
            if st.button("Check Outliers"):
              try:
                with st.spinner("正在检测异常值..."):
                    df = st.session_state.processed_data1.copy()
                    z_scores = df[selected_cols].apply(zscore)

                    if isinstance(z_scores, pd.Series):
                       z_scores = z_scores.to_frame()

                    outliers = (z_scores.abs() > threshold).any(axis=1)
                    num_outliers = outliers.sum()

                    st.session_state.outliers = outliers  
                    st.session_state.num_outliers = num_outliers

                    st.warning(f"Find {num_outliers} Outliers!")
                    
                    st.dataframe(df[outliers].head(10), height=200)
              except:
                  st.warning('请先fill blanks')
                    
            if st.button("Remove Outliers", key="remove_outliers"):
                try:
                        df = st.session_state.processed_data1.copy()
                        outliers = st.session_state.outliers
                        df_cleaned = df[~outliers]
                        st.session_state.processed_data1 = df_cleaned
                        st.session_state.outliers_removed = True  # 设置状态标志
                        st.success(f"Droped {st.session_state.num_outliers} Outliers, Remaining {len(df_cleaned)} columns！")
                        if  st.session_state.get("outliers_removed", False): 
                           st.subheader("Brief View After Cleaning")
                           st.dataframe(st.session_state.processed_data1.head(10), height=150)
                except: 
                        st.warning('请先check outliers!')
                       
                if st.session_state.processed_data1 is not None:

                # 创建一个内存缓冲区
                 output = BytesIO()
    
                # 使用 Pandas 的 ExcelWriter 和 xlsxwriter 引擎写入 Excel
                 with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.processed_data1.to_excel(writer, index=False, sheet_name='Filled_Data')
    
               # 获取 Excel 文件的二进制内容
                 processed_excel = output.getvalue()

              # 添加下载按钮
                 st.download_button(
        label="Download cleaned Data as Excel",
        data=processed_excel,
        file_name="filled_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
            

    
    show_navigation()

#-------------------------------------步骤二 数据编码和标准化---------------------------------
#------------------------------------------------------------------------------------------

elif st.session_state.current_step == 2:
    st.markdown('<h1 style="font-size:50px; color:gray;">Step2: Data Reprosessing</h1>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data1
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    encoding_method = st.radio("Choose Encoding method", ["Label 标签编码（推荐）", "One-Hot 独热编码"], horizontal=True)
    
    scaling_method = st.radio("Choose Scaling Method", ["None 不缩放", "Standardize 标准化", "Normalize 归一化"], horizontal=True)
    
    if st.button("应用转换"):
        with st.spinner("处理中..."):
            processed_df = df.copy()
            if categorical_cols:
                processed_df.loc[processed_df['生育政策'] == '家庭补助金', '生育政策'] = '税务优惠'
                processed_df.loc[processed_df['生育政策'] == '带薪产假', '生育政策'] = '育儿补贴'
            if categorical_cols:
                if encoding_method == "One-Hot 独热编码":
                    encoder = OneHotEncoder(sparse=False)
                    encoded = encoder.fit_transform(processed_df[categorical_cols])
                    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
                    processed_df = pd.concat([processed_df.drop(categorical_cols, axis=1), encoded_df], axis=1)
                else:
                    for col in categorical_cols:
                        le = LabelEncoder()
                        processed_df[col] = le.fit_transform(processed_df[col])
            if scaling_method != "None 不缩放" and numeric_cols:
                scaler = StandardScaler() if scaling_method == "Standardize 标准化" else MinMaxScaler()
                processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
            st.session_state.processed_data2 = processed_df
            st.success("数据处理完成！")
            st.rerun()
    
    if st.session_state.processed_data2 is not None:
        st.subheader("Brief View After Reposessing")
        st.dataframe(st.session_state.processed_data2.head(10), height=150)

                # 创建一个内存缓冲区
        output = BytesIO()
    
                # 使用 Pandas 的 ExcelWriter 和 xlsxwriter 引擎写入 Excel
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.processed_data1.to_excel(writer, index=False, sheet_name='Filled_Data')
    
               # 获取 Excel 文件的二进制内容
        processed_excel = output.getvalue()

              # 添加下载按钮
        st.download_button(
        label="Download possessed Data as Excel",
        data=processed_excel,
        file_name="filled_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    show_navigation()

#--------------------------------------第三步：特征选择与模型训练---------------------------------
#--------------------------------------------------------------------------------------------

elif st.session_state.current_step == 3:
    st.markdown('<h1 style="font-size:50px; color:gray;">Step3: Model Training</h1>', unsafe_allow_html=True)

    df = st.session_state.processed_data2
    if df is None:
        st.error("数据未加载，请返回上一步")
        st.stop()

    # 特征选择列
    col1, col2 = st.columns([2, 3])
    with col1:
        target = st.selectbox(
            "Choose Objective",
            options=df.columns,
            index=df.columns.get_loc('人口增长率') if '人口增长率' in df.columns else len(df.columns) - 1
        )

        available_features = [col for col in df.columns if col != target]
        default_features = [col for col in available_features if col not in ['国家', '年份', '地区']]

        features = st.multiselect(
            "Choose Features",
            options=available_features,
            default=default_features
        )
        st.session_state.features = features
        st.session_state.target = target

    # 模型训练列
    with col2:
        model_type = st.selectbox(
            "Choose Model Type",
            options=["LinearRegression 线性回归", "DecisionTree 决策树", "RandomForest 随机森林", "KNeighbors KNN回归", "SVM 支持向量机"],
            index=0
        )
        train_size = st.slider("Train-Test Split", 0.5, 0.9, 0.8, step=0.05)

        # 超参数调优
        tuning_enabled = st.checkbox("调参")

        if tuning_enabled:
            if model_type == "DecisionTree 决策树":
                param_grid = {
                    'max_depth': st.slider("Max Depth", 1, 20, (3, 10)),
                    'min_samples_split': st.slider("Min Samples Split", 2, 10, (2, 5))
                }
            elif model_type == "RandomForest 随机森林":
                param_grid = {
                    'n_estimators': st.slider("Number of Trees", 10, 200, (50, 100)),
                    'max_depth': st.slider("Max Depth", 1, 20, (5, 15))
                }
            elif model_type == "KNeighbors KNN回归":
                param_grid = {
                    'n_neighbors': st.slider("Number of Neighbors", 1, 20, (3, 5)),
                    'weights': st.selectbox("Weights", options=["uniform", "distance"])
                }
            elif model_type == "SVM 支持向量机":
                param_grid = {
                    'C': st.slider("C (Regularization)", 0.1, 10.0, (1.0, 2.0), step=0.1),
                    'kernel': st.selectbox("Kernel", options=["linear", "poly", "rbf", "sigmoid"])
                }
            else:
                st.warning("该模型不适用调参")
                param_grid = None

        # 训练模型
        if st.button("Train!", use_container_width=True):
            with st.spinner("训练中..."):
                try:
                    X = df[st.session_state.features]
                    y = df[target]
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
                        "SVM 支持向量机": SVR()
                    }
                    model = model_map[model_type]

                    # 超参数调优
                    if tuning_enabled and param_grid:
                        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=1)
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                        st.success(f"最佳超参数: {best_params}")
                    else:
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

#----------------------------------------第四步：可视化分析------------------------------------
#-------------------------------------------------------------------------------------------

elif st.session_state.current_step == 4:
    st.markdown('<h1 style="font-size:50px; color:gray;">Step4: Data Visualization</h1>', unsafe_allow_html=True)
    
    if not st.session_state.get('model'):
        st.error("模型未训练，请返回第三步")
        st.stop()
    
    #模型性能
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
    
    #画图
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

#----------------------------------------第五步：预测新数据------------------------------------
#-------------------------------------------------------------------------------------------

elif st.session_state.current_step == 5:
    st.markdown('<h1 style="font-size:50px; color:gray;">Step5: Model Prediction</h1>', unsafe_allow_html=True)
    
    if not st.session_state.get('model'):
        st.error("模型未训练，请返回第三步")
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
    

#----------------------------------------第六步：数据智能分析-------------------------------------
#---------------------------------------------------------------------------------------------
    
elif st.session_state.current_step == 6:
    st.markdown('<h1 style="font-size:50px; color:gray;">Step6: Data Analysis</h1>', unsafe_allow_html=True)
    # 初始化对话历史记录
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # 检查必要状态
    if not st.session_state.get('model'):
        st.error("模型未训练，请返回第三步")
        st.stop()

    greating = 'Hi! I am your stat1016 assistant and I can give you analysis based on the \ndata you just produced.'
    
    if st.session_state.n0 == 0:
     placeholder = st.empty()  # 创建一个占位符
     displayed_text = ""       # 用于逐步显示的变量
     st.session_state.n0 = 1
     for char in greating:
                   displayed_text += char
                   placeholder.markdown(f"```markdown\n{displayed_text}\n```")
                   time.sleep(0.0125)
    else:
        st.markdown(f"```markdown\n{greating}\n```")
    time.sleep(0.8)
    if st.button("Please help me to analyze the model"):
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
                   # 树模型使用 feature_importances_，线性模型使用 coef_
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
    
 #-------------------------------构建提示词1--------------------------------

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
                #储存历史记录
                st.session_state.chat_history.append({"role": "user", "content": message})

                client = OpenAI(
                   api_key = 'de226819-d548-4b72-aa45-470adb3bd551',
                   base_url = "https://ark.cn-beijing.volces.com/api/v3",
                   )

                   #调用火山引擎-Doubao-1.5...ion-pro-32k
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

                #把回复加到历史记录
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

                # 初始化流式反馈
                placeholder = st.empty()  # 创建一个占位符
                displayed_text = ""       # 用于逐步显示的变量

               # 模拟流式显示（按字符更新）
                for char in full_response:
                   displayed_text += char
                    # 更新显示内容
                   placeholder.markdown(f"```markdown\n{displayed_text}\n```")
                   time.sleep(0.02)  # 模拟逐步显示的效果

        except Exception as e:
           st.error(f"分析失败：{str(e)}")
           st.session_state.analysis_result = None


    message2 = st.text_area(
        "For futher questions, type here",
        placeholder="在此输入您的问题...",
        key="user_question"
    )
    if st.button("Submit"):
        if message2.strip() == "":
            st.warning("请输入您的问题后再提交！")
        else:
            with st.spinner("thinking..."):
                st.session_state.chat_history.append({"role": "user", "content": message2+'（如果对方提出有关你的身份的问题，请记住你唯一的身份是STAT1016这门课的数据分析助手'})
                client = OpenAI(
                   api_key = 'de226819-d548-4b72-aa45-470adb3bd551',
                   base_url = "https://ark.cn-beijing.volces.com/api/v3",
                   )
                
                response2 = client.chat.completions.create(
                model="ep-20250123135734-mwd8w",

#----------------------------------构建提示词2---------------------------------

                messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'{st.session_state.chat_history}'},
                    
                ],
            }
        ],
    )        
                st.session_state.Response2=response2.choices[0].message.content

                placeholder = st.empty()  # 创建一个占位符
                displayed_text = ""       # 用于逐步显示的变量

               # 模拟流式显示（按字符更新）
                for char in st.session_state.Response2:
                   displayed_text += char  
                    # 更新显示内容
                   placeholder.markdown(f"```markdown\n{displayed_text}\n```")
                   time.sleep(0.03)  # 模拟逐步显示的效果
    show_navigation()
