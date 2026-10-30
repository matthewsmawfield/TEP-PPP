function loadJsonIfExists(path) {
    if (typeof require === 'undefined') {
        return null;
    }
    const fs = require('fs');
    if (!fs.existsSync(path)) {
        return null;
    }
    return JSON.parse(fs.readFileSync(path, 'utf8'));
}

function num(v, fallback) {
    if (v === undefined || v === null || (typeof v === 'number' && !isFinite(v))) {
        return fallback !== undefined ? fallback : null;
    }
    return v;
}

function fmt(v, decimals) {
    const n = num(v);
    if (n === null) return 'N/A';
    if (decimals !== undefined) {
        return Number(n).toFixed(decimals);
    }
    return String(n);
}

function createManuscriptContext() {
    if (typeof require === 'undefined') {
        return {};
    }
    const path = require('path');
    const projectRoot = path.join(__dirname, '..');

    // Load step result JSONs from the streamlined 4-step pipeline
    const step3Summary = loadJsonIfExists(path.join(projectRoot, 'results', 'outputs', 'step_03_empirical_gradient.json')) || {};
    const step4Summary = loadJsonIfExists(path.join(projectRoot, 'results', 'outputs', 'step_04_variance_reduction.json')) || {};

    const context = {};

    // Step 03: Multi-day phase-coherent spatial correlation
    context['STEP3_N_STATIONS'] = fmt(num(step3Summary.n_stations), 0);
    context['STEP3_N_PAIRS'] = fmt(num(step3Summary.n_pairs), 0);
    context['STEP3_N_DAYS'] = fmt(num(step3Summary.n_days), 0);
    context['STEP3_N_PAIR_DAYS'] = fmt(num(step3Summary.n_pair_days), 0);
    context['STEP3_MEAN_PA'] = fmt(num(step3Summary.mean_pa_all_days), 3);
    context['STEP3_MC_Z_SCORE'] = fmt(num(step3Summary.mc_z_score), 2);
    context['STEP3_MC_P_VALUE'] = fmt(num(step3Summary.mc_p_value), 4);
    context['STEP3_LAMBDA_T_KM'] = fmt(num(step3Summary.lambda_t_km), 0);
    context['STEP3_FIT_R2'] = fmt(num(step3Summary.fit_r2), 3);
    context['STEP3_DAILY_CONSISTENCY'] = fmt(num(step3Summary.daily_consistency_std), 3);

    // Step 04: Multi-day EW/NS anisotropy and CMB alignment
    context['STEP4_N_PAIRS'] = fmt(num(step4Summary.n_pairs), 0);
    context['STEP4_EW_MEAN_DIST'] = fmt(num(step4Summary.ew_mean_dist_km), 0);
    context['STEP4_NS_MEAN_DIST'] = fmt(num(step4Summary.ns_mean_dist_km), 0);
    context['STEP4_EW_NS_DIST_RATIO'] = fmt(num(step4Summary.ew_ns_distance_ratio), 2);
    context['STEP4_DIST_PA_CORR'] = fmt(num(step4Summary.distance_pa_correlation), 3);
    context['STEP4_EW_MEAN_PA_NAIVE'] = fmt(num(step4Summary.ew_mean_pa_naive), 3);
    context['STEP4_NS_MEAN_PA_NAIVE'] = fmt(num(step4Summary.ns_mean_pa_naive), 3);
    context['STEP4_BEST_FIT_RA'] = fmt(num(step4Summary.best_fit_ra), 0);
    context['STEP4_BEST_FIT_DEC'] = fmt(num(step4Summary.best_fit_dec), 0);
    context['STEP4_BEST_FIT_CORR'] = fmt(num(step4Summary.best_fit_partial_corr), 3);
    context['STEP4_CMB_PARTIAL_CORR'] = fmt(num(step4Summary.cmb_partial_corr), 3);
    context['STEP4_CMB_Z_SCORE'] = fmt(num(step4Summary.cmb_z_score), 2);
    context['STEP4_CMB_P_VALUE'] = fmt(num(step4Summary.cmb_p_value), 4);
    context['STEP4_ANGULAR_SEP_DEG'] = fmt(num(step4Summary.angular_sep_deg), 0);

    return context;
}

function injectPlaceholders(template, context = createManuscriptContext()) {
    return template.replace(/\{\{\s*([A-Za-z0-9_]+)\s*\}\}/g, (match, key) => {
        const value = context[key];
        if (value === undefined || value === null) {
            return match;
        }
        return String(value);
    });
}

if (typeof module !== 'undefined') {
    module.exports = { createManuscriptContext, injectPlaceholders };
}
