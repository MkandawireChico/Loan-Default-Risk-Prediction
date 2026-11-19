import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import Papa from "papaparse";
import {
  Activity, AlertCircle, BarChart3, Bell,
  Building2, CheckCircle2, ChevronDown, ChevronRight, CircleHelp,
  ClipboardList, Cloud, Database, Download, FileSpreadsheet, Gauge,
  LayoutDashboard, Menu, RefreshCw, Settings2, ShieldCheck,
  SlidersHorizontal, Sparkles, Users, X, Zap,
} from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
type Page = "overview" | "assessment" | "batch" | "governance";
type Summary = {
  applications: number; defaults: number; default_rate: number;
  total_requested_exposure_mwk: number; average_loan_amount_mwk: number;
  median_monthly_income_mwk: number; average_loan_to_income_ratio: number;
};
type Group = { name: string; applications: number; defaults: number; default_rate: number; exposure_mwk: number };
type ModelInfo = { model_version: string; model_type: string; feature_count: number; numeric_features: number; categorical_features: number; excluded_demographic_fields: string[]; review_threshold: number; data_notice: string; use_restriction: string };
type Contribution = { feature: string; contribution_log_odds: number };
type Assessment = { model_version: string; probability_of_default: number; risk_band: string; review_flag: boolean; interpretation: string; disclaimer: string; explanation?: { method: string; scale: string; risk_increasing: Contribution[]; risk_decreasing: Contribution[]; warning: string } };
type GroupAuditRow = { group: string; sample_size: number; suppressed: boolean; observed_default_rate?: number; model_review_rate?: number; recall?: number | null; false_positive_rate?: number | null; precision?: number | null };
type EvalReport = { subgroup_diagnostics?: { minimum_group_size: number; groups: Record<string, GroupAuditRow[]>; interpretation: string }; final_test?: { sample_size: number; default_rate: number; roc_auc: number; pr_auc: number; brier_score: number; log_loss: number; threshold: number; precision: number; recall: number; f1: number; specificity: number; review_rate: number; confusion_matrix: Record<string, number> }; threshold_selection?: { review_capacity: number; selected_validation_operating_point: { threshold: number; precision: number; recall: number; review_rate: number } }; limitations?: string[] };
type RegistryVersion = { model_version: string; model_type: string; status: string; artifact_path: string; artifact_sha256: string; dataset_sha256: string; training_rows: number | null; training_seed: number | null; decision_threshold: number | null; registered_at: string; final_test: Record<string, number> };
type RegistryResponse = { active_version: string; versions: RegistryVersion[]; data_notice: string };

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json() as Promise<T>;
}
async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data as T;
}
const money = (value: number) => new Intl.NumberFormat("en-MW", { style: "currency", currency: "MWK", maximumFractionDigits: 0 }).format(value);
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [breakdowns, setBreakdowns] = useState<{ by_branch: Group[]; by_customer_segment: Group[]; by_loan_purpose: Group[] } | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [evaluation, setEvaluation] = useState<EvalReport | null>(null);
  const [registry, setRegistry] = useState<RegistryResponse | null>(null);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [mobileNav, setMobileNav] = useState(false);
  const [sample, setSample] = useState<Record<string, unknown> | null>(null);
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [assessing, setAssessing] = useState(false);
  const [batchResults, setBatchResults] = useState<Record<string, unknown> | null>(null);
  const [batchBusy, setBatchBusy] = useState(false);
  const [batchError, setBatchError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true); setError("");
    try {
      const [health, s, b, m, e, r] = await Promise.all([
        getJson<{ status: string }>("/health"),
        getJson<Summary>("/api/v1/portfolio/summary"),
        getJson<typeof breakdowns>("/api/v1/portfolio/breakdowns"),
        getJson<ModelInfo>("/api/v1/model"),
        getJson<EvalReport>("/api/v1/model/evaluation"),
        getJson<RegistryResponse>("/api/v1/model/registry"),
      ]);
      setApiStatus(health.status === "ok" ? "online" : "offline");
      setSummary(s); setBreakdowns(b); setModel(m); setEvaluation(e); setRegistry(r);
    } catch (err) {
      setApiStatus("offline"); setError(err instanceof Error ? err.message : "Unable to load platform data.");
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const loadSample = useCallback(async () => {
    try {
      const data = await getJson<{ features: Record<string, unknown> }>("/api/v1/assessments/sample");
      setSample(data.features); setAssessment(null);
    } catch (err) { setError(err instanceof Error ? err.message : "Unable to load sample."); }
  }, []);

  useEffect(() => { if (page === "assessment" && !sample) void loadSample(); }, [page, sample, loadSample]);

  const navItems = [
    { id: "overview" as Page, label: "Portfolio overview", icon: LayoutDashboard, group: "WORKSPACE" },
    { id: "assessment" as Page, label: "Application assessment", icon: ClipboardList, group: "WORKSPACE" },
    { id: "batch" as Page, label: "Batch analysis", icon: FileSpreadsheet, group: "WORKSPACE" },
    { id: "governance" as Page, label: "Model governance", icon: ShieldCheck, group: "INTELLIGENCE" },
  ];

  const changeFeature = (name: string, value: string) => {
    if (!sample) return;
    const numeric = ["Monthly_Income", "Loan_Amount", "Loan_Term_Months", "Interest_Rate", "Credit_Score", "Existing_Debt", "Savings_Balance", "Previous_Default_Count", "Number_of_Late_Payments", "Existing_Loan_Count"];
    const next: Record<string, unknown> = { ...sample, [name]: numeric.includes(name) ? Number(value) : value };
    const income = Number(next.Monthly_Income) || 0;
    const loan = Number(next.Loan_Amount) || 0;
    const term = Math.max(1, Number(next.Loan_Term_Months) || 1);
    const annualRate = (Number(next.Interest_Rate) || 0) / 100;
    const monthlyRate = annualRate / 12;
    const factor = Math.pow(1 + monthlyRate, term);
    const repayment = monthlyRate === 0 ? loan / term : loan * monthlyRate * factor / (factor - 1);
    const safeAnnualIncome = income * 12;
    const repaymentRatio = income > 0 ? repayment / income : 0;
    const savings = Number(next.Savings_Balance) || 0;
    const creditScore = Number(next.Credit_Score) || 0;
    const previousDefaults = Number(next.Previous_Default_Count) || 0;
    const latePayments = Number(next.Number_of_Late_Payments) || 0;
    const creditHistory = Number(next.Credit_History_Length_Months) || 0;
    next.Debt_to_Income_Ratio = safeAnnualIncome > 0 ? ((Number(next.Existing_Debt) || 0) + loan * 0.1) / safeAnnualIncome : null;
    next.Loan_to_Income_Ratio = safeAnnualIncome > 0 ? loan / safeAnnualIncome : null;
    next.Monthly_Repayment_Ratio = repaymentRatio;
    next.Savings_to_Income_Ratio = safeAnnualIncome > 0 ? savings / safeAnnualIncome : null;
    next.Previous_Default_Flag = previousDefaults > 0 ? 1 : 0;
    next.Late_Payment_Frequency = creditHistory > 0 ? latePayments / creditHistory : null;
    next.Loan_Burden_Category = repaymentRatio < 0.20 ? "Low" : repaymentRatio < 0.35 ? "Moderate" : repaymentRatio < 0.50 ? "High" : "Severe";
    next.Credit_Risk_Category = creditScore < 580 ? "Poor" : creditScore < 670 ? "Fair" : creditScore < 740 ? "Good" : "Very Good";
    setSample(next);
    setAssessment(null);
  };

  const runAssessment = async () => {
    if (!sample) return;
    setAssessing(true); setError("");
    try { setAssessment(await postJson<Assessment>("/api/v1/assessments/explain", { features: sample })); }
    catch (err) { setError(err instanceof Error ? err.message : "Assessment failed."); }
    finally { setAssessing(false); }
  };

  const handleBatch = (file: File) => {
    setBatchError(""); setBatchResults(null); setBatchBusy(true);
    Papa.parse<Record<string, unknown>>(file, {
      header: true, skipEmptyLines: true, dynamicTyping: true,
      complete: async (parsed) => {
        try {
          if (!parsed.data.length) throw new Error("The CSV contains no application rows.");
          if (parsed.data.length > 500) throw new Error("Batch limit is 500 applications per request.");
          const result = await postJson<Record<string, unknown>>("/api/v1/assessments/batch", { applications: parsed.data });
          setBatchResults(result);
        } catch (err) { setBatchError(err instanceof Error ? err.message : "Batch scoring failed."); }
        finally { setBatchBusy(false); }
      },
      error: (err) => { setBatchError(err.message); setBatchBusy(false); },
    });
  };

  const downloadBatch = () => {
    const results = (batchResults?.results || []) as Array<Record<string, unknown>>;
    if (!results.length) return;
    const csv = Papa.unparse(results);
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8;" }));
    const link = document.createElement("a"); link.href = url; link.download = "nyanja-credit-risk-batch-results.csv"; link.click(); URL.revokeObjectURL(url);
  };

  const branchData = useMemo(() => (breakdowns?.by_branch || []).slice(0, 8).map((item) => ({ ...item, label: item.name.length > 11 ? item.name.slice(0, 10) + "…" : item.name })), [breakdowns]);
  const segmentData = breakdowns?.by_customer_segment || [];
  const pieColors = ["#6d8df5", "#37b6a0", "#e5a64e", "#a78bfa", "#e47785"];

  return (
    <div className="app-shell">
      {mobileNav && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-mark"><Activity size={20} strokeWidth={2.4} /></div>
          <div><div className="brand-name">NYANJA</div><div className="brand-sub">CREDIT INTELLIGENCE</div></div>
          <button className="icon-button close-nav" onClick={() => setMobileNav(false)} aria-label="Close menu"><X size={18}/></button>
        </div>
        <div className="workspace-switch"><div className="workspace-avatar">NC</div><div className="workspace-copy"><strong>Nyanja Commercial</strong><span>Research workspace</span></div><ChevronDown size={15}/></div>
        <div className="nav-label">WORKSPACE</div>
        <nav>{navItems.filter(n => n.group === "WORKSPACE").map(item => <button key={item.id} className={`nav-item ${page === item.id ? "active" : ""}`} onClick={() => { setPage(item.id); setMobileNav(false); }}><item.icon size={18}/><span>{item.label}</span>{page === item.id && <span className="nav-active-dot"/>}</button>)}</nav>
        <div className="nav-label nav-label-spaced">INTELLIGENCE</div>
        <nav>{navItems.filter(n => n.group === "INTELLIGENCE").map(item => <button key={item.id} className={`nav-item ${page === item.id ? "active" : ""}`} onClick={() => { setPage(item.id); setMobileNav(false); }}><item.icon size={18}/><span>{item.label}</span></button>)}</nav>
        <div className="sidebar-bottom">
          <div className="help-card"><div className="help-icon"><CircleHelp size={17}/></div><strong>Need a hand?</strong><span>Explore the platform guide and model documentation.</span><a href="https://github.com/MkandawireChico/Loan-Default-Risk-Prediction" target="_blank" rel="noreferrer">View documentation <ChevronRight size={13}/></a></div>
          <div className="profile-row"><div className="profile-avatar">CM</div><div className="profile-copy"><strong>Credit analyst</strong><span>Demo account</span></div><Settings2 size={17}/></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left"><button className="icon-button menu-button" onClick={() => setMobileNav(true)} aria-label="Open menu"><Menu size={20}/></button><div className="breadcrumb">Workspace <ChevronRight size={14}/> <strong>{navItems.find(n => n.id === page)?.label}</strong></div></div>
          <div className="topbar-actions"><div className={`connection-pill ${apiStatus}`}><span className="status-dot"/>{apiStatus === "online" ? "API connected" : apiStatus === "checking" ? "Connecting" : "API offline"}</div><button className="icon-button" title="Refresh data" onClick={() => void refresh()}><RefreshCw size={17}/></button><button className="icon-button notification-button" title="Notifications"><Bell size={17}/><i/></button><div className="topbar-avatar">CM</div></div>
        </header>

        <div className="content-wrap">
          <div className="notice-banner"><div className="notice-icon"><AlertCircle size={16}/></div><p><strong>Research simulation.</strong> All borrower records and outcomes are synthetic. This platform is for education and demonstration, not real lending decisions.</p><a href="https://github.com/MkandawireChico/Loan-Default-Risk-Prediction/blob/feature/credit-risk-platform/docs/PLATFORM_ROADMAP.md" target="_blank" rel="noreferrer">Project scope <ChevronRight size={13}/></a></div>
          {error && <div className="error-banner"><AlertCircle size={16}/>{error}<button onClick={() => setError("")}><X size={15}/></button></div>}

          {page === "overview" && <section>
            <div className="page-heading"><div><div className="eyebrow">PORTFOLIO INTELLIGENCE <span className="eyebrow-line"/></div><h1>Portfolio overview</h1><p>Monitor portfolio composition, exposure and synthetic default patterns.</p></div><div className="heading-actions"><span className="data-tag"><Database size={14}/> Synthetic dataset</span><button className="secondary-button" onClick={() => void refresh()}><RefreshCw size={15}/> Refresh</button></div></div>
            {loading ? <div className="loading-panel"><div className="spinner"/><span>Loading portfolio intelligence…</span></div> : summary ? <>
              <div className="kpi-grid">
                <Kpi title="Loan applications" value={summary.applications.toLocaleString()} sub="Synthetic records" icon={<ClipboardList size={18}/>} tone="blue" />
                <Kpi title="Observed default rate" value={percent(summary.default_rate)} sub={`${summary.defaults.toLocaleString()} simulated defaults`} icon={<Activity size={18}/>} tone="rose" />
                <Kpi title="Requested exposure" value={money(summary.total_requested_exposure_mwk)} sub="Total requested loan value" icon={<Building2 size={18}/>} tone="violet" />
                <Kpi title="Average loan amount" value={money(summary.average_loan_amount_mwk)} sub="Per application" icon={<Gauge size={18}/>} tone="amber" />
              </div>
              <div className="insight-grid">
                <div className="panel chart-panel"><div className="panel-heading"><div><h3>Default rate by branch</h3><p>Observed outcomes across synthetic branch locations</p></div><span className="panel-icon"><BarChart3 size={17}/></span></div>
                  <div className="chart-area">{branchData.length ? <ResponsiveContainer width="100%" height="100%"><BarChart data={branchData} margin={{top:12,right:8,left:-18,bottom:0}}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e9edf3"/><XAxis dataKey="label" axisLine={false} tickLine={false} tick={{fontSize:11,fill:"#778397"}}/><YAxis axisLine={false} tickLine={false} tickFormatter={(v) => `${Math.round(v*100)}%`} tick={{fontSize:11,fill:"#778397"}}/><Tooltip formatter={(v) => percent(Number(v))} contentStyle={{border:"1px solid #e8edf3",borderRadius:10,fontSize:12}}/><Bar dataKey="default_rate" radius={[5,5,0,0]} maxBarSize={42}>{branchData.map((_,i)=><Cell key={i} fill={pieColors[i%pieColors.length]}/>)}</Bar></BarChart></ResponsiveContainer> : <Empty/>}</div>
                  <div className="chart-footnote">Branch differences reflect the synthetic generation process. They are not real-world regional findings.</div>
                </div>
                <div className="panel segment-panel"><div className="panel-heading"><div><h3>Customer segments</h3><p>Application distribution</p></div><span className="panel-icon"><Users size={17}/></span></div>
                  <div className="donut-wrap">{segmentData.length ? <><div className="donut-chart"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={segmentData} dataKey="applications" nameKey="name" innerRadius={58} outerRadius={83} paddingAngle={3} stroke="none">{segmentData.map((_,i)=><Cell key={i} fill={pieColors[i%pieColors.length]}/>)}</Pie><Tooltip formatter={(v) => Number(v).toLocaleString()}/></PieChart></ResponsiveContainer><div className="donut-center"><strong>{summary.applications.toLocaleString()}</strong><span>Applications</span></div></div><div className="legend-list">{segmentData.map((s,i)=><div className="legend-row" key={s.name}><i style={{background:pieColors[i%pieColors.length]}}/><span>{s.name}</span><strong>{s.applications.toLocaleString()}</strong></div>)}</div></> : <Empty/>}</div>
                </div>
              </div>
              <div className="panel table-panel"><div className="panel-heading"><div><h3>Branch portfolio breakdown</h3><p>Application volumes, observed default rates and requested exposure</p></div><span className="table-count">{breakdowns?.by_branch.length || 0} branches</span></div><div className="table-scroll"><table><thead><tr><th>Branch</th><th>Applications</th><th>Defaults</th><th>Default rate</th><th>Requested exposure</th></tr></thead><tbody>{(breakdowns?.by_branch || []).map(row=><tr key={row.name}><td><span className="branch-name"><span className="branch-dot"/>{row.name}</span></td><td>{row.applications.toLocaleString()}</td><td>{row.defaults.toLocaleString()}</td><td><span className="rate-cell">{percent(row.default_rate)}</span></td><td>{money(row.exposure_mwk)}</td></tr>)}</tbody></table></div></div>
              <div className="secondary-insights"><div className="mini-stat"><span>Median monthly income</span><strong>{money(summary.median_monthly_income_mwk)}</strong><small>Across synthetic applications</small></div><div className="mini-stat"><span>Average loan-to-income ratio</span><strong>{summary.average_loan_to_income_ratio.toFixed(2)}×</strong><small>Loan amount relative to monthly income</small></div><div className="mini-stat"><span>Model version</span><strong className="model-version">{model?.model_version || "—"}</strong><small>Current inference artifact</small></div></div>
            </> : <div className="empty-state"><AlertCircle/><strong>Portfolio data unavailable</strong><span>Check the API service and try again.</span><button className="primary-button" onClick={() => void refresh()}>Retry connection</button></div>}
          </section>}

          {page === "assessment" && <section>
            <div className="page-heading"><div><div className="eyebrow">DECISION SUPPORT <span className="eyebrow-line"/></div><h1>Application assessment</h1><p>Explore a calibrated probability estimate using a synthetic application profile.</p></div><button className="secondary-button" onClick={() => void loadSample()}><RefreshCw size={15}/> Load another sample</button></div>
            <div className="assessment-layout"><div className="panel assessment-form"><div className="form-heading"><div><h3>Applicant and facility profile</h3><p>Adjust selected fields. Other model inputs are prefilled from a synthetic record.</p></div><span className="sample-badge"><Sparkles size={13}/> Demo profile</span></div>
              {sample ? <><div className="form-section-title">INCOME & FACILITY</div><div className="form-grid">
                <Field label="Monthly income (MWK)" value={sample.Monthly_Income} onChange={v=>changeFeature("Monthly_Income",v)} />
                <Field label="Loan amount (MWK)" value={sample.Loan_Amount} onChange={v=>changeFeature("Loan_Amount",v)} />
                <Field label="Loan term (months)" value={sample.Loan_Term_Months} onChange={v=>changeFeature("Loan_Term_Months",v)} />
                <Field label="Interest rate (%)" value={sample.Interest_Rate} onChange={v=>changeFeature("Interest_Rate",v)} />
                <Field label="Existing debt (MWK)" value={sample.Existing_Debt} onChange={v=>changeFeature("Existing_Debt",v)} />
                <Field label="Savings balance (MWK)" value={sample.Savings_Balance} onChange={v=>changeFeature("Savings_Balance",v)} />
              </div><div className="form-section-title">CREDIT HISTORY</div><div className="form-grid">
                <Field label="Credit score" value={sample.Credit_Score} onChange={v=>changeFeature("Credit_Score",v)} />
                <Field label="Previous defaults" value={sample.Previous_Default_Count} onChange={v=>changeFeature("Previous_Default_Count",v)} />
                <Field label="Late payments" value={sample.Number_of_Late_Payments} onChange={v=>changeFeature("Number_of_Late_Payments",v)} />
                <Field label="Existing loans" value={sample.Existing_Loan_Count} onChange={v=>changeFeature("Existing_Loan_Count",v)} />
              </div><div className="form-actions"><span><ShieldCheck size={15}/> Demographic fields excluded from model inputs</span><button className="primary-button" onClick={() => void runAssessment()} disabled={assessing}>{assessing ? <><span className="button-spinner"/> Assessing…</> : <>Run assessment <ChevronRight size={16}/></>}</button></div></> : <div className="loading-panel"><div className="spinner"/><span>Loading synthetic application…</span></div>}
            </div><div className="assessment-result"><div className="result-head"><div><div className="eyebrow">MODEL OUTPUT</div><h3>Risk assessment</h3></div><span className="result-icon"><Zap size={18}/></span></div>
              {assessment ? <div className="result-content"><div className="probability-label">ESTIMATED PROBABILITY OF DEFAULT</div><div className="probability-value">{percent(assessment.probability_of_default)}</div><div className="probability-track"><span style={{width:percent(assessment.probability_of_default)}}/></div><div className={`risk-pill risk-${assessment.risk_band.toLowerCase()}`}>{assessment.risk_band} risk band</div><div className="result-detail"><span>Model review flag</span><strong>{assessment.review_flag ? "Flagged for review" : "Below review threshold"}</strong></div><p className="result-interpretation">{assessment.interpretation}</p>{assessment.explanation && <div className="explanation-block"><div className="explanation-title">Top model contributors <span>Log-odds scale</span></div><div className="contribution-heading positive-heading">Increasing the model score</div>{assessment.explanation.risk_increasing.map(item=><div className="contribution-row" key={item.feature}><span>{item.feature.replace(/^(numeric|categorical)__/, "").replace(/_/g, " ")}</span><strong>+{item.contribution_log_odds.toFixed(3)}</strong></div>)}<div className="contribution-heading negative-heading">Reducing the model score</div>{assessment.explanation.risk_decreasing.map(item=><div className="contribution-row" key={item.feature}><span>{item.feature.replace(/^(numeric|categorical)__/, "").replace(/_/g, " ")}</span><strong>{item.contribution_log_odds.toFixed(3)}</strong></div>)}<p className="explanation-warning">{assessment.explanation.warning}</p></div>}<div className="result-disclaimer"><AlertCircle size={15}/>{assessment.disclaimer}</div></div> : <div className="result-placeholder"><div className="placeholder-orbit"><Gauge size={29}/></div><strong>Your assessment will appear here</strong><span>Adjust the synthetic profile and run an assessment to see the model output.</span></div>}
              <div className="result-model"><span>MODEL VERSION</span><strong>{model?.model_version || "Loading…"}</strong></div>
            </div></div>
          </section>}

          {page === "batch" && <section>
            <div className="page-heading"><div><div className="eyebrow">PORTFOLIO OPERATIONS <span className="eyebrow-line"/></div><h1>Batch analysis</h1><p>Score multiple synthetic application records in one request.</p></div><span className="data-tag"><SlidersHorizontal size={14}/> Up to 500 rows</span></div>
            <div className="batch-layout"><div className="panel upload-panel"><div className="upload-icon"><FileSpreadsheet size={25}/></div><h3>Upload application data</h3><p>Choose a CSV file containing the model's 33 feature columns. The API validates each row and reports invalid records.</p><label className="upload-drop"><input type="file" accept=".csv,text/csv" onChange={e=>{const f=e.target.files?.[0];if(f)handleBatch(f);}}/><Cloud size={24}/><strong>Choose a CSV file</strong><span>or drag a file here · CSV up to 500 rows</span></label>{batchBusy && <div className="upload-progress"><div className="spinner"/><span>Scoring applications…</span></div>}{batchError && <div className="error-inline"><AlertCircle size={15}/>{batchError}</div>}<div className="template-note"><CircleHelp size={15}/><span>Required headers must match the feature schema exactly. Identifiers, target labels and demographic audit fields are not accepted.</span></div></div>
              <div className="panel batch-guide"><h3>Batch workflow</h3><div className="guide-step"><span>01</span><div><strong>Prepare the file</strong><p>Use the exact API feature names as CSV headers.</p></div></div><div className="guide-step"><span>02</span><div><strong>Validate and score</strong><p>Each row is validated. Invalid rows are returned with an error message.</p></div></div><div className="guide-step"><span>03</span><div><strong>Review results</strong><p>Download the scored records for further analysis.</p></div></div><div className="guide-warning"><AlertCircle size={16}/> Do not upload personal, customer or real banking data.</div></div></div>
            {batchResults && <div className="panel batch-results"><div className="panel-heading"><div><h3>Batch scoring results</h3><p>Model: {String(batchResults.model_version)}</p></div><button className="primary-button" onClick={downloadBatch}><Download size={15}/> Download results</button></div><div className="batch-kpis"><Kpi title="Submitted" value={String(batchResults.submitted)} sub="Input rows" icon={<FileSpreadsheet size={17}/>} tone="blue"/><Kpi title="Scored" value={String(batchResults.scored)} sub="Valid rows" icon={<CheckCircle2 size={17}/>} tone="green"/><Kpi title="Rejected" value={String(batchResults.rejected)} sub="Rows requiring correction" icon={<AlertCircle size={17}/>} tone="amber"/></div><div className="table-scroll"><table><thead><tr><th>Row</th><th>Probability of default</th><th>Risk band</th><th>Review flag</th></tr></thead><tbody>{((batchResults.results || []) as Array<Record<string,unknown>>).slice(0,20).map(row=><tr key={String(row.row)}><td>{String(row.row)}</td><td>{percent(Number(row.probability_of_default))}</td><td>{String(row.risk_band)}</td><td>{row.review_flag ? "Review" : "Not flagged"}</td></tr>)}</tbody></table></div></div>}
          </section>}

          {page === "governance" && <section>
            <div className="page-heading"><div><div className="eyebrow">MODEL RISK MANAGEMENT <span className="eyebrow-line"/></div><h1>Model governance</h1><p>Inspect held-out performance, calibration context and model limitations.</p></div><span className="governance-tag"><ShieldCheck size={14}/> Research model</span></div>
            {evaluation?.final_test ? <><div className="model-summary-card"><div className="model-summary-main"><span className="model-chip"><Sparkles size={13}/> CURRENT MODEL</span><h2>{model?.model_type || "Calibrated Logistic Regression"}</h2><p>{model?.model_version} · {evaluation.final_test.sample_size.toLocaleString()} held-out test records</p></div><div className="model-summary-side"><span>Review threshold</span><strong>{percent(evaluation.final_test.threshold)}</strong><small>Selected using validation data</small></div></div>
              <div className="metric-grid"><Metric title="ROC-AUC" value={evaluation.final_test.roc_auc.toFixed(3)} detail="Ranking discrimination" /><Metric title="PR-AUC" value={evaluation.final_test.pr_auc.toFixed(3)} detail="Precision-recall performance" /><Metric title="Brier score" value={evaluation.final_test.brier_score.toFixed(3)} detail="Probability accuracy · lower is better" /><Metric title="Log loss" value={evaluation.final_test.log_loss.toFixed(3)} detail="Probabilistic error · lower is better" /></div>
              <div className="governance-grid"><div className="panel"><div className="panel-heading"><div><h3>Operating point</h3><p>Final test metrics at the validation-selected threshold</p></div></div><div className="operating-list"><MetricRow label="Precision" value={percent(evaluation.final_test.precision)}/><MetricRow label="Recall" value={percent(evaluation.final_test.recall)}/><MetricRow label="Specificity" value={percent(evaluation.final_test.specificity)}/><MetricRow label="F1 score" value={evaluation.final_test.f1.toFixed(3)}/><MetricRow label="Review rate" value={percent(evaluation.final_test.review_rate)}/></div></div><div className="panel"><div className="panel-heading"><div><h3>Confusion matrix</h3><p>Held-out classification counts</p></div></div><div className="confusion-grid"><div className="confusion-cell tn"><span>True negatives</span><strong>{evaluation.final_test.confusion_matrix.true_negatives.toLocaleString()}</strong></div><div className="confusion-cell fp"><span>False positives</span><strong>{evaluation.final_test.confusion_matrix.false_positives.toLocaleString()}</strong></div><div className="confusion-cell fn"><span>False negatives</span><strong>{evaluation.final_test.confusion_matrix.false_negatives.toLocaleString()}</strong></div><div className="confusion-cell tp"><span>True positives</span><strong>{evaluation.final_test.confusion_matrix.true_positives.toLocaleString()}</strong></div></div></div></div>
              <div className="panel governance-panel"><div className="panel-heading"><div><h3>Feature governance</h3><p>Demographic attributes excluded from predictive model inputs</p></div><ShieldCheck size={19} className="green-icon"/></div><div className="audit-tags">{(model?.excluded_demographic_fields || []).map(f=><span key={f}>{f.replace(/_/g," ")}</span>)}</div><p className="governance-copy">These fields remain in the synthetic dataset for descriptive and post-hoc fairness analysis. Excluding direct demographic fields does not guarantee fairness. Proxy variables and differences in data quality still require investigation.</p></div>
              {evaluation.subgroup_diagnostics && <div className="panel governance-panel"><div className="panel-heading"><div><h3>Subgroup performance diagnostics</h3><p>Observed outcomes and model error rates across selected groups</p></div><Users size={18} className="green-icon"/></div><div className="table-scroll"><table><thead><tr><th>Dimension</th><th>Group</th><th>Sample</th><th>Observed default</th><th>Model review rate</th><th>Recall</th><th>False positive rate</th></tr></thead><tbody>{Object.entries(evaluation.subgroup_diagnostics.groups).flatMap(([dimension, rows]) => rows.map((row, index) => <tr key={dimension + row.group}><td>{index === 0 ? dimension.replace(/_/g, " ") : ""}</td><td>{row.group}</td><td>{row.sample_size}</td><td>{row.suppressed ? "Suppressed" : percent(row.observed_default_rate || 0)}</td><td>{row.suppressed ? "—" : percent(row.model_review_rate || 0)}</td><td>{row.suppressed || row.recall == null ? "—" : percent(row.recall)}</td><td>{row.suppressed || row.false_positive_rate == null ? "—" : percent(row.false_positive_rate)}</td></tr>))}</tbody></table></div><p className="governance-copy">{evaluation.subgroup_diagnostics.interpretation} Minimum group size: {evaluation.subgroup_diagnostics.minimum_group_size}.</p></div>}
              <div className="panel governance-panel"><div className="panel-heading"><div><h3>Model registry</h3><p>Version history, training provenance and artifact integrity fingerprints</p></div><Database size={18} className="green-icon"/></div>{registry?.versions.length ? <div className="table-scroll"><table><thead><tr><th>Version</th><th>Status</th><th>Model family</th><th>Training rows</th><th>Threshold</th><th>Test ROC-AUC</th></tr></thead><tbody>{registry.versions.map(item=><tr key={item.model_version}><td><strong>{item.model_version}</strong><div className="registry-hash">Artifact {item.artifact_sha256.slice(0, 12)}…</div><div className="registry-hash">Dataset {item.dataset_sha256.slice(0, 12)}…</div></td><td><span className={item.status === "active" ? "risk-pill risk-low" : "risk-pill risk-moderate"}>{item.status}</span></td><td>{item.model_type}</td><td>{item.training_rows?.toLocaleString() || "—"}</td><td>{item.decision_threshold == null ? "—" : percent(item.decision_threshold)}</td><td>{item.final_test?.roc_auc == null ? "—" : item.final_test.roc_auc.toFixed(3)}</td></tr>)}</tbody></table></div> : <Empty/>}<p className="governance-copy">The registry stores model metadata and SHA-256 fingerprints. Model files remain in the application image. A matching fingerprint identifies the exact artifact and source dataset used for this registered version. {registry?.data_notice}</p></div>
              <div className="panel limitations-panel"><div className="panel-heading"><div><h3>Known limitations</h3><p>Read before interpreting any model result</p></div><AlertCircle size={18} className="warning-icon"/></div><ul>{(evaluation.limitations || []).map(item=><li key={item}>{item}</li>)}</ul></div>
            </> : <div className="empty-state"><AlertCircle/><strong>Evaluation report unavailable</strong><span>Build and train the v2 model to publish held-out evaluation results.</span></div>}
          </section>}
          <footer className="app-footer"><span>NYANJA CREDIT RISK INTELLIGENCE <b>·</b> SYNTHETIC RESEARCH PLATFORM</span><span>Built for responsible ML experimentation</span></footer>
        </div>
      </main>
    </div>
  );
}

function Kpi({title,value,sub,icon,tone}:{title:string;value:string;sub:string;icon:ReactNode;tone:string}) {
  return <div className="kpi-card"><div className="kpi-top"><span>{title}</span><span className={`kpi-icon ${tone}`}>{icon}</span></div><strong className="kpi-value">{value}</strong><span className="kpi-sub">{sub}</span></div>;
}
function Field({label,value,onChange}:{label:string;value:unknown;onChange:(v:string)=>void}) {
  return <label className="field"><span>{label}</span><input type="number" value={String(value ?? "")} onChange={e=>onChange(e.target.value)}/></label>;
}
function Metric({title,value,detail}:{title:string;value:string;detail:string}) {
  return <div className="metric-card"><span>{title}</span><strong>{value}</strong><small>{detail}</small></div>;
}
function MetricRow({label,value}:{label:string;value:string}) {
  return <div className="metric-row"><span>{label}</span><strong>{value}</strong></div>;
}
function Empty() { return <div className="chart-empty">No portfolio data available.</div>; }
