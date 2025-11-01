/* apps/dashboard/static/js/dashboard.js */

// Flask 템플릿의 url_for('dashboard.model_performance')와 같은 Flask 변수는
// 이 외부 JS 파일에서는 직접 사용할 수 없습니다.
// 따라서, HTML 템플릿에서 URL을 전역 변수로 설정하거나,
// fetch 함수가 인라인 스크립트로 유지되도록 HTML 파일을 수정해야 합니다.
// 여기서는 HTML 파일에서 URL을 전달받는 방식으로 구조를 변경합니다.

// 임시 함수 (실제 Flask 환경에서는 HTML에서 설정한 전역 URL 변수를 사용해야 합니다.)
// 예: fetch(window.DASHBOARD_URLS.model_performance);
// 이 코드에서는 Flask URL을 직접 사용할 수 없으므로,
// URL을 인수로 받는 함수로 구조를 변경하겠습니다.

document.addEventListener('DOMContentLoaded', function() {
    // URL을 직접 인수로 받기 위해 로컬에서만 사용하는 더미 함수를 사용합니다.
    // 실제 Flask 앱에서는 HTML <script> 블록에서 이 함수를 호출할 때
    // Flask url_for() 값을 인수로 넘겨야 합니다.
    console.log("MLOps Dashboard scripts loaded.");

    // 클래스 이름에 따른 색상 매핑 함수
    function get_color_by_class(className) {
        const colors = {
            'setosa': 'rgb(255, 99, 132)',
            'versicolor': 'rgb(54, 162, 235)',
            'virginica': 'rgb(75, 192, 192)',
        };
        return colors[className] || 'rgb(128, 128, 128)';
    }


    // --- 1. AI 모델 성능 시각화 ---
    async function fetchPerformanceData(url) {
        try {
            const response = await fetch(url);
            const data = await response.json();
            
            // 지표 업데이트
            document.getElementById('accuracy_metric').innerText = data.metrics.accuracy;
            document.getElementById('precision_metric').innerText = data.metrics.precision;
            document.getElementById('recall_metric').innerText = data.metrics.recall;
            document.getElementById('f1_score_metric').innerText = data.metrics.f1_score;

            // 새로운 AUC 지표 업데이트
            document.getElementById('macro_auc_metric').innerText = data.metrics.auc_roc_macro.toFixed(4);
            document.getElementById('micro_auc_metric').innerText = data.metrics.auc_roc_micro.toFixed(4);
            document.getElementById('weighted_auc_metric').innerText = data.metrics.auc_roc_weighted.toFixed(4);

            // ROC Curve
            const rocCtx = document.getElementById('rocChart').getContext('2d');
            const rocDatasets = [];
            for (const className in data.roc_data.fpr) {
                const color = get_color_by_class(className);
                rocDatasets.push({
                    label: `ROC (${className}) AUC: ${data.roc_data.auc[className].toFixed(2)}`,
                    data: data.roc_data.tpr[className].map((tpr, i) => ({x: data.roc_data.fpr[className][i], y: tpr})),
                    borderColor: color,
                    backgroundColor: color,
                    borderWidth: 2,
                    fill: false
                });
            }
            // 기존 Chart 인스턴스가 있다면 파괴
            if (window.rocChartInstance) {
                window.rocChartInstance.destroy();
            }
            window.rocChartInstance = new Chart(rocCtx, {
                type: 'line',
                data: { datasets: rocDatasets },
                options: {
                    responsive: true,
                    aspectRatio: 1, // ROC 차트 비율을 1:1로 설정
                    scales: { x: { type: 'linear', position: 'bottom', title: { display: true, text: 'False Positive Rate' } }, y: { title: { display: true, text: 'True Positive Rate' } } }
                }
            });

            // PR Curve
            const prCtx = document.getElementById('prChart').getContext('2d');
            const prDatasets = [];
            for (const className in data.pr_data.precision) {
                const color = get_color_by_class(className);
                prDatasets.push({
                    label: `PR (${className}) AUC: ${data.pr_data.auc[className].toFixed(2)}`,
                    data: data.pr_data.precision[className].map((precision, i) => ({x: data.pr_data.recall[className][i], y: precision})),
                    borderColor: color,
                    backgroundColor: color,
                    borderWidth: 2,
                    fill: false
                });
            }
            if (window.prChartInstance) {
                window.prChartInstance.destroy();
            }
            window.prChartInstance = new Chart(prCtx, {
                type: 'line',
                data: { datasets: prDatasets },
                options: {
                    responsive: true,
                    aspectRatio: 1, // PR 차트 비율을 1:1로 설정
                    scales: { x: { type: 'linear', position: 'bottom', title: { display: true, text: 'Recall' } }, y: { title: { display: true, text: 'Precision' } } }
                }
            });

            // 혼동 행렬 테이블 생성
            const cmData = data.confusion_matrix;
            if (cmData && cmData.matrix.length > 0) {
                const cmContainer = document.getElementById('confusion_matrix_container');
                const table = document.createElement('table');
                table.className = 'table table-bordered text-center';
                
                const thead = table.createTHead();
                const headerRow = thead.insertRow();
                headerRow.innerHTML = '<th></th>'; // 빈 좌측 상단 셀
                cmData.labels.forEach(label => {
                    const th = document.createElement('th');
                    th.innerText = `예측: ${label}`;
                    headerRow.appendChild(th);
                });
                
                const tbody = table.createTBody();
                cmData.matrix.forEach((row, i) => {
                    const tr = tbody.insertRow();
                    const rowHeader = document.createElement('th');
                    rowHeader.innerText = `실제: ${cmData.labels[i]}`;
                    tr.appendChild(rowHeader);
                    row.forEach((cell, j) => {
                        const td = document.createElement('td');
                        td.innerText = cell;
                        if (i === j) {
                            td.classList.add('table-success'); // 정답인 경우 배경색
                        } else {
                            td.classList.add('table-danger'); // 오답인 경우 배경색
                        }
                        tr.appendChild(td);
                    });
                });
                
                cmContainer.innerHTML = '';
                cmContainer.appendChild(table);
            }

        } catch (error) {
            console.error("Error fetching performance data:", error);
            document.getElementById('performance_metrics').innerHTML = `<p class="text-danger">모델 성능 데이터를 불러오지 못했습니다. (URL: ${url})</p>`;
        }
    }

    // --- 2. 데이터 드리프트 시각화 ---
    async function fetchDataDriftData(url) {
        try {
            // Fetch API를 사용하여 JSON 데이터를 가져옵니다.
            const response = await fetch(url);
            // HTTP 상태 코드가 200 범위가 아닐 경우 에러 처리
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            
            const driftStatus = document.getElementById('drift_status');
            const driftChartContainer = document.getElementById('drift_chart_container');
            driftChartContainer.innerHTML = '';
            driftStatus.innerHTML = ''; // 상태 메시지 초기화
            
            let allFine = true;
            
            // 데이터 드리프트 감지 결과를 표시합니다.
            if (Object.keys(data.drift_results).length === 0) {
                 driftStatus.innerHTML = `<div class="alert alert-info py-1"><strong>데이터가 부족하여 드리프트를 감지할 수 없습니다.</strong></div>`;
                 return;
            }


            for (const feature in data.drift_results) {
                const result = data.drift_results[feature];
                
                // Python에서 정수로 변환되었으므로 1인지 확인합니다.
                const isDriftDetected = result.drift_detected === 1; 
                
                const alertClass = isDriftDetected ? 'alert-danger' : 'alert-success';
                const message = `<strong>${feature}:</strong> ${result.message} (p-value: ${result.p_value})`;
                driftStatus.innerHTML += `<div class="alert ${alertClass} py-1">${message}</div>`;
                
                if (isDriftDetected) {
                    allFine = false;
                }
            }
            
            if (allFine && Object.keys(data.drift_results).length > 0) {
                 // 모든 특성이 정상일 경우 가장 상위에 종합 메시지를 추가
                 const allFineMessage = `<div class="alert alert-success py-1"><strong>모든 특성에서 데이터 드리프트가 감지되지 않았습니다.</strong></div>`;
                 driftStatus.innerHTML = allFineMessage + driftStatus.innerHTML;
            }

            // Plotly.js를 이용한 히스토그램 시각화 (type: 'histogram' 활용)
            const features = Object.keys(data.drift_data);
            if (features.length > 0) {
                features.forEach(feature => {
                    // Bootstrap grid item (col)으로 감쌉니다.
                    const colDiv = document.createElement('div');
                    colDiv.className = 'col'; 

                    const plotDiv = document.createElement('div');
                    plotDiv.id = `drift_chart_${feature}`;
                    // Plotly 차트가 삽입될 div. 반응형을 위해 높이만 지정
                    plotDiv.style.height = '400px'; 
                    plotDiv.className = 'bg-white rounded-lg shadow p-2'; 
                    
                    colDiv.appendChild(plotDiv);
                    driftChartContainer.appendChild(colDiv);

                    const refData = data.drift_data[feature].reference;
                    const currData = data.drift_data[feature].current;

                    const traceRef = {
                        x: refData, // 원시 데이터 전달
                        name: '기준 데이터 (과거 30일)',
                        type: 'histogram', // Plotly.js에서 직접 빈을 계산
                        opacity: 0.6,
                        marker: { color: 'rgba(54, 162, 235, 1)' } // 파란색 계열
                    };
                    const traceCurr = {
                        x: currData, // 원시 데이터 전달
                        name: '최근 데이터 (최근 7일)',
                        type: 'histogram', // Plotly.js에서 직접 빈을 계산
                        opacity: 0.6,
                        marker: { color: 'rgba(255, 99, 132, 1)' } // 빨간색 계열
                    };

                    const layout = {
                        barmode: 'overlay', // 오버레이 모드로 겹쳐서 표시
                        title: `<b>${feature}</b> 분포 비교`,
                        xaxis: { 
                            title: feature,
                            automargin: true
                        },
                        yaxis: { 
                            title: '빈도',
                            automargin: true 
                        },
                        margin: { t: 30, r: 10, b: 70, l: 30 }, // 차트 여백 조정 b:70
                        responsive: true
                    };

                    Plotly.newPlot(plotDiv.id, [traceRef, traceCurr], layout);
                });
            }
        } catch (error) {
            console.error("Error fetching drift data:", error);
            // 오류 메시지를 사용자에게 표시
            document.getElementById('drift_status').innerHTML = `<div class="alert alert-danger">데이터 드리프트 데이터를 불러오지 못했습니다. 서버 오류: ${error.message} (URL: ${url})</div>`;
        }
    }
    
    // --- 3. 예측 결과 모니터링 시각화 ---
    async function fetchPredictionMonitorData(url) {
        try {
            const response = await fetch(url);
            const data = await response.json();
            
            const ctx = document.getElementById('predictionMonitorChart').getContext('2d');
            if (window.predMonitorChartInstance) {
                window.predMonitorChartInstance.destroy();
            }
            window.predMonitorChartInstance = new Chart(ctx, {
                type: 'line',
                data: data.data,
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'top' },
                        title: { display: true, text: '시간에 따른 예측 결과 분포' }
                    },
                    scales: { x: { type: 'category' }, y: { stacked: true, title: { display: true, text: '예측 건수' } } }
                }
            });
        } catch (error) {
            console.error("Error fetching prediction monitor data:", error);
            const container = document.getElementById('predictionMonitorChart').parentElement;
            container.innerHTML = `<div class="alert alert-danger">예측 모니터링 데이터를 불러오지 못했습니다. (URL: ${url})</div>`;
        }
    }

    // --- 4. 이상 감지 및 알림 ---
    async function fetchAnomalyData(url) {
        try {
            const response = await fetch(url);
            const data = await response.json();
            
            const alertsContainer = document.getElementById('anomaly_alerts');
            alertsContainer.innerHTML = '';
            
            if (data.anomalies.length > 0) {
                data.anomalies.forEach(anomaly => {
                    const alert = document.createElement('div');
                    alert.className = 'alert alert-danger py-2';
                    alert.innerHTML = `
                        <strong>이상 감지!</strong> ID: #${anomaly.id}<br>
                        특성: ${anomaly.features}<br>
                        예측: ${anomaly.predicted_class}
                    `;
                    alertsContainer.appendChild(alert);
                });
            } else {
                alertsContainer.innerHTML = `<div class="alert alert-success">최근 예측에서 이상 데이터가 감지되지 않았습니다.</div>`;
            }
        } catch (error) {
            console.error("Error fetching anomaly data:", error);
            document.getElementById('anomaly_alerts').innerHTML = `<div class="alert alert-warning">이상 감지 데이터를 불러오지 못했습니다. (URL: ${url})</div>`;
        }
    }

    // --- 5. 리소스 사용량 (더미) ---
    function updateResourceUsage() {
        const cpuUsage = Math.floor(Math.random() * 50) + 30; // 30-80%
        const memUsage = Math.floor(Math.random() * 40) + 50; // 50-90%
        
        const cpuBar = document.getElementById('cpu_usage_bar');
        cpuBar.style.width = cpuUsage + '%';
        cpuBar.innerText = cpuUsage + '%';
        
        const memBar = document.getElementById('mem_usage_bar');
        memBar.style.width = memUsage + '%';
        memBar.innerText = memUsage + '%';
    }

    // 전역으로 노출된 URL 객체에서 URL을 가져와 데이터 로딩 함수 실행
    if (window.DASHBOARD_URLS) {
        fetchPerformanceData(window.DASHBOARD_URLS.model_performance);
        fetchDataDriftData(window.DASHBOARD_URLS.data_drift);
        fetchPredictionMonitorData(window.DASHBOARD_URLS.prediction_monitor);
        fetchAnomalyData(window.DASHBOARD_URLS.anomaly_detection);
    } else {
        console.error("DASHBOARD_URLS object not found. Ensure it is defined in the HTML template.");
    }
    
    // 주기적 업데이트
    updateResourceUsage();
    setInterval(updateResourceUsage, 5000);
});