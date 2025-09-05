import logging
import numpy as np
import pandas as pd
from sklearn import metrics
from transformers import set_seed


def normalize_score(score, method='none', min=None, max=None, mean=None, std=None, match_mean=None, match_std=None):
    # score: np.ndarray, shape: (n_probes, n_galleries)\
    if method == 'minmax':
        if min is None or max is None:
            normed_score = (np.nan_to_num(score) - np.nan_to_num(score).min()) / (
                        np.nan_to_num(score).max() - np.nan_to_num(score).min() + 1e-10)
        else:
            normed_score = (np.nan_to_num(score) - min) / (max - min + 1e-10)
    elif method == 'zscore':
        if mean is None or std is None:
            normed_score = (np.nan_to_num(score) - np.nan_to_num(score).mean()) / (np.nan_to_num(score).std() + 1e-10)
        else:
            normed_score = (np.nan_to_num(score) - mean) / (std + 1e-10)
    elif method == 'rhe':
        """
        Perform Reduction of High-scores Effect (RHE) normalization.
        He, Mingxing, et al. "Performance evaluation of score level fusion in multimodal biometric systems." Pattern Recognition 43.5 (2010): 1789-1800.
        """
        assert min is not None or match_mean is not None or match_std is not None, "min, match_mean, or match_std is required for RHE normalization"
        normed_score = (np.nan_to_num(score) - min) / (match_mean + match_std - min + 1e-10)
    elif method == 'none':
        normed_score = np.nan_to_num(score)
    else:
        raise NotImplementedError(f'{method} not implemented')
    return normed_score


def score_fusion_method(score_mats, method='none'):
    """
    score_mats: list of numpy arrays, shape: (n_probes, n_galleries)
    """
    # Ensure score_mats is a numpy array
    score_mats = np.array(score_mats)  # Convert list of arrays to a numpy array if necessary

    if method == 'min':
        # min score fusion
        # Jain, Anil, Karthik Nandakumar, and Arun Ross. "Score normalization in multimodal biometric systems." Pattern recognition 38.12 (2005): 2270-2285.
        return np.min(score_mats, axis=0)  # Compute the minimum score across all score mats
    elif method == 'mean':
        # mean score fusion
        return np.mean(score_mats, axis=0)  # Compute the mean score across all score mats
    elif method == 'max':
        # max score fusion
        return np.max(score_mats, axis=0)  # Compute the maximum score across all score mats
    else:
        raise NotImplementedError(f'{method} not implemented')


def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=1e-8):
    assert len(score_vec.shape) == 1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == bool
    score_neg = score_vec[~label_vec]
    score_neg = np.sort(score_neg)[::-1]  # score from high to low
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        epsilon = 1e-5
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0] + epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1] - epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = (num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm == 0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm - 1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds


def DIR_FAR(score_mat, label_mat, ranks=[1], FARs=[1.0], get_retrievals=False):
    ''' Closed/Open-set Identification.
        A general case of Cummulative Match Characteristic (CMC)
        where thresholding is allowed for open-set identification.
    args:
        score_mat (np.array): a P x G matrix, P is number of probes, G is size of gallery
        label_mat:            a P x G matrix, bool
        ranks:                a list of integers
        FARs:                 false alarm rates, if 1.0, closed-set identification (CMC)
        get_retrievals:       not implemented yet
    return:
        DIRs:                 an F x R matrix, F is the number of FARs, R is the number of ranks,
                              flatten into a vector if F=1 or R=1.
        FARs:                 an vector of length = F.
        thredholds:           an vector of length = F.
    '''
    assert score_mat.shape == label_mat.shape
    assert np.all(label_mat.astype(np.float32).sum(axis=1) <= 1)
    # Split the matrix for match probes and non-match probes
    # subfix _m: match, _nm: non-match
    # For closed set, we only use the match probes
    mate_indices = label_mat.astype(bool).any(axis=1)
    score_mat_m = score_mat[mate_indices, :]
    label_mat_m = label_mat[mate_indices, :]
    score_mat_nm = score_mat[np.logical_not(mate_indices), :]
    label_mat_nm = label_mat[np.logical_not(mate_indices), :]
    mate_indices = np.argwhere(mate_indices).flatten()

    # print('mate probes: %d, non mate probes: %d' % (score_mat_m.shape[0], score_mat_nm.shape[0]))

    # Find the thresholds for different FARs
    max_score_nm = np.max(score_mat_nm, axis=1)
    label_temp = np.zeros(max_score_nm.shape, dtype=bool)
    if len(FARs) == 1 and FARs[0] >= 1.0:
        # If only testing closed-set identification, use the minimum score as thrnp.vstack((eshold
        # in case there is no non-mate probes
        thresholds = [np.min(score_mat) - 1e-10]
    else:
        # If there is open-set identification, find the thresholds by FARs.
        assert score_mat_nm.shape[
                   0] > 0, "For open-set identification (FAR<1.0), there should be at least one non-mate probe!"
        thresholds = find_thresholds_by_FAR(max_score_nm, label_temp, FARs=FARs)

    # ingredients for score plot
    mate_scores = score_mat_m[label_mat_m]
    nonmate_maxscores = max_score_nm
    plot_labels = [1] * len(mate_scores) + [0] * len(nonmate_maxscores)
    scoreplot_df = pd.DataFrame({'scores': np.concatenate([mate_scores, nonmate_maxscores]), 'labels': plot_labels})
    idx_scoreplot_df = pd.DataFrame({'indv_nonmate_scores': score_mat_nm.flatten()})

    # Sort the labels row by row according to scores
    sort_idx_mat_m = np.argsort(score_mat_m, axis=1)[:, ::-1]
    sorted_label_mat_m = np.ndarray(label_mat_m.shape, dtype=bool)
    sorted_score_mat_m = score_mat_m.copy()
    for row in range(label_mat_m.shape[0]):
        sort_idx = (sort_idx_mat_m[row, :])
        sorted_label_mat_m[row, :] = label_mat_m[row, sort_idx]
        sorted_score_mat_m[row, :] = score_mat_m[row, sort_idx]

    # Calculate DIRs for different FARs and ranks
    gt_score_m = score_mat_m[label_mat_m]
    assert gt_score_m.size == score_mat_m.shape[0]

    DIRs = np.zeros([len(FARs), len(ranks)], dtype=np.float32)
    FARs = np.zeros([len(FARs)], dtype=np.float32)
    success = np.ndarray((len(FARs), len(ranks)), dtype=object)
    for i, threshold in enumerate(thresholds):
        for j, rank in enumerate(ranks):
            score_rank = gt_score_m >= threshold
            retrieval_rank = sorted_label_mat_m[:, 0:rank].any(axis=1)
            DIRs[i, j] = (score_rank & retrieval_rank).astype(np.float32).mean()
            if get_retrievals:
                success[i, j] = (score_rank & retrieval_rank)
        if score_mat_nm.shape[0] > 0:
            FARs[i] = (max_score_nm >= threshold).astype(np.float32).mean()

    if DIRs.shape[0] == 1 or DIRs.shape[1] == 1:
        DIRs = DIRs.flatten()
        success = success.flatten()

    if get_retrievals:
        return DIRs, FARs, thresholds, mate_indices, success, sort_idx_mat_m, sorted_score_mat_m

    return DIRs, FARs, thresholds, scoreplot_df, idx_scoreplot_df


def compute_ap_cmc(index, good_index, junk_index):
    """ Compute AP and CMC for each sample
    """
    ap = 0
    cmc = np.zeros(len(index)) 
    
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        ap = ap + d_recall*precision

    return ap, cmc


def compute_tar_at_far(similarity_matrix, q_pids, g_pids, far=[0.0001, 0.001, 0.01]):
    """
    Calculate the True Accept Rate (TAR) at a specified False Accept Rate (FAR) and the corresponding threshold.
    
    Args:
    score_mat: numpy array, size P x G, where P is the number of probes and G is the number of gallery entries.
    q_pids: numpy array, size P, the labels for the probes.
    g_pids: numpy array, size G, the labels for the gallery entries.
    far: float, desired FAR value (default is 0.01).

    Returns:
    tar: float, the TAR value at the specified FAR.
    far_threshold: float, the threshold corresponding to the specified FAR.
    """
    
    # Generate a mask matrix to indicate matches (1) and non-matches (0)
    mask_matrix = q_pids.reshape(-1, 1) == g_pids.reshape(1, -1)
    
    # Flatten the score matrix and the mask matrix
    scores = similarity_matrix.flatten()
    mask = mask_matrix.flatten()

    # Handle NaN values: treat NaNs as failures by setting them to the lowest score
    nan_mask = np.isnan(scores)
    if (~nan_mask).sum() > 0:
        scores[nan_mask] = scores[~nan_mask].min()
    else:
        scores[nan_mask] = 0.0
    
    # Roc_curve to compute FAR and TAR
    fpr, tpr, thresholds = metrics.roc_curve(mask, scores)
    
    tar_list = []
    far_threshold_list = []  
    for k in far:
        # Find the index of the closest threshold that matches the desired FAR
        idx = np.where(fpr >= k)[0][0]
            
        tar = tpr[idx]       # TAR at the specified FAR
        far_threshold = thresholds[idx]  # Score threshold corresponding to the specified FAR
        tar_list.append(tar)
        far_threshold_list.append(far_threshold)
    
    tar_result = pd.Series(tar_list, index=[f"TAR@{k:.2%}FAR" for k in far])
    far_threshold = pd.Series(far_threshold_list, index=[f"Threshold@{k:.2%}FAR" for k in far])
    
    result = pd.concat([tar_result, far_threshold], axis=0)
    return result


def compute_fnir_at_fpir(similarity_matrix, q_pids, g_pids, g1_pids, ranks=[1], fars=[0.0001, 0.001, 0.01]):
    """
    Calculate the False Negative Identification Rate (FNIR) at a specified False Positive Identification Rate (FPIR) and the corresponding threshold.
    
    Args:
    score_mat: numpy array, size P x G, where P is the number of probes and G is the number of gallery entries.
    q_pids: numpy array, size P, the labels for the probes.
    g_pids: numpy array, size G, the labels for the whole gallery entries.
    g1_pids: numpy array, size G1, the labels for the subset of g_pids. MUST BE SORTED!
    far: float, desired FAR value (default is 0.01).

    Returns:
    fnir: float, the FNIR value at the specified FPIR.
    fpir_threshold: float, the threshold corresponding to the specified FPIR.
    """
    # reconstruct the similarity matrix to include only the subset of g_pids
    scores1 = similarity_matrix[:, np.isin(g_pids, g1_pids)]
    
    q_pids = q_pids.reshape(-1, 1)
    g1_pids = g1_pids.reshape(-1, 1)
    
    DIRs_openset, _, openset_thresholds, scoreplot_df, idx_scoreplot_df = DIR_FAR(scores1,
                                                                            q_pids == g1_pids.T,
                                                                            ranks=ranks,
                                                                            FARs=fars)

    # DIRs_openset1, _, thresholds1, scoreplot_df1, idx_scoreplot_df1 = DIR_FAR(scores1,
    #                                                                         q_pids == g1_pids.T,
    #                                                                         FARs=far)
    # DIRs_openset2, _, thresholds2, scoreplot_df2, idx_scoreplot_df2 = DIR_FAR(scores2,
    #                                                                         q_pids == g2_pids.T,
    #                                                                         FARs=far)
    # DIRs_openset = (DIRs_openset1 + DIRs_openset2) / 2.0
    # openset_thresholds = (thresholds1 + thresholds2) / 2.0
    # scoreplot_df = pd.concat([scoreplot_df1, scoreplot_df2], axis=0)
    # idx_scoreplot_df = pd.concat([idx_scoreplot_df1, idx_scoreplot_df2], axis=0)

    open_result1 = pd.Series(DIRs_openset, index=[f"TPIR@{k:.2%}FPIR" for k in fars])
    open_result2 = pd.Series(1 - DIRs_openset, index=[f"FNIR@{k:.2%}FPIR" for k in fars])
    open_threshold = pd.Series(openset_thresholds,
                                index=[f"FNIR@{k:.2%}FPIR_threshold" for k in fars])
    
    result = pd.concat([open_result1, open_result2, open_threshold], axis=0)
    
    # compute statistics
    num_probe_templates = len(q_pids)
    num_probe_subjects = len(np.unique(q_pids))
    num_gallery_subjects = len(g_pids)
    num_galley1_templates = len(g1_pids)
    num_gallary1_subjects = len(np.unique(g1_pids))
    # num_galley2_templates = len(g2_pids)
    # num_gallary2_subjects = len(np.unique(g2_pids))
    result = pd.concat([result, pd.Series({'num_probe_templates': num_probe_templates,
                                           'num_probe_subjects': num_probe_subjects,
                                           'num_gallery_subjects': num_gallery_subjects,
                                           'num_galley1_templates': num_galley1_templates,
                                           'num_gallary1_subjects': num_gallary1_subjects,}),
                        ])
    return result


def evaluate(similarity_matrix, q_pids, g_pids, q_camids, g_camids):
    """ Compute CMC and mAP

    Args:
        similarity_matrix (numpy ndarray): similarity matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
    """
    num_q, num_g = similarity_matrix.shape
    index = np.argsort(-similarity_matrix, axis=1) # from large to small

    num_no_gt = 0 # num of query imgs without groundtruth
    no_gt_index=[] # index of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0
    
    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            no_gt_index.append(i)
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        logger = logging.getLogger('reid.evaluate')
        logger.info("{} query samples do not have groundtruth.".format(num_no_gt))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)
    
    return CMC, mAP


def evaluate_with_clothes(similarity_matrix, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, mode='CC'):
    """ Compute CMC and mAP with clothes

    Args:
        similarity_matrix (numpy ndarray): similarity matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
        q_clothids (numpy array): clothes IDs for query samples.
        g_clothids (numpy array): clothes IDs for gallery samples.
        mode: 'CC' for clothes-changing; 'SC' for the same clothes.
    """
    assert mode in ['CC', 'SC']
    
    num_q, num_g = similarity_matrix.shape
    index = np.argsort(-similarity_matrix, axis=1) # from large to small

    num_no_gt = 0 # num of query imgs without groundtruth
    no_gt_index=[] # index of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        cloth_index = np.argwhere(g_clothids==q_clothids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if mode == 'CC':
            good_index = np.setdiff1d(good_index, cloth_index, assume_unique=True)
            # remove gallery samples that have the same (pid, camid) or (pid, clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.intersect1d(query_index, cloth_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        else:
            good_index = np.intersect1d(good_index, cloth_index)
            # remove gallery samples that have the same (pid, camid) or 
            # (the same pid and different clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.setdiff1d(query_index, cloth_index)
            junk_index = np.union1d(junk_index1, junk_index2)

        if good_index.size == 0:
            num_no_gt += 1
            no_gt_index.append(i)
            continue
    
        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        logger = logging.getLogger('reid.evaluate')
        logger.info("{} query samples do not have groundtruth.".format(num_no_gt))

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0

    return CMC, mAP


def evaluate_evp(similarity_matrix, q_pids, g_pids, far=[0.0001, 0.001, 0.01]):
    """ 
    Compute TAR@FAR (Verification), Ranking and FNIR@FPIR (Open-Search) metrics on Biometric recogntion task

    Args:
        similarity_matrix (numpy ndarray): similarity matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples/tracklets. NOTE: it can have repeated IDs, 

    """
    num_q, num_g = similarity_matrix.shape
    index = np.argsort(-similarity_matrix, axis=1) # from large to small

    num_no_gt = 0 # num of query imgs without groundtruth
    no_gt_index=[] # index of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        good_index = query_index
        if good_index.size == 0:
            num_no_gt += 1
            no_gt_index.append(i)
            continue
    
        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, [])
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        logger = logging.getLogger('reid.evaluate')
        logger.info("{} query samples do not have groundtruth.".format(num_no_gt))

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0
        
    # Compute TAR@FAR
    verification_result = compute_tar_at_far(similarity_matrix, q_pids, g_pids, far)
    # Compute FNIR@FPIR
    openset_pids = np.random.choice(g_pids, 130, replace=False)
    opensearch_result = compute_fnir_at_fpir(similarity_matrix, q_pids, g_pids, openset_pids, far)
    
    cmc_index = [0, 4, 9, 19]
    cmc_result = pd.Series([CMC[i] for i in cmc_index], index=[f"Rank{i+1}" for i in cmc_index])
    mAP_result = pd.Series([mAP], index=['mAP'])
    result = pd.concat([cmc_result, mAP_result, verification_result, opensearch_result], axis=0)
    return result


def test_score(score_mat, merge_score_mat, 
               q_pids, q_camids, q_clothes_ids, 
               g_pids, g_camids, g_clothes_ids, unique_g_pids, 
               dataset='ccvid', log=False, seed=None):
    """
    Args:
        score_mat (np.array): similarity matrix with shape (num_query, num_gallery).
        merge_score_mat (np.array): similarity matrix with shape (num_query, num_gallery).
        q_pids (np.array): person IDs for query samples.
        q_camids (np.array): camera IDs for query samples.
        q_clothes_ids (np.array): clothes IDs for query samples.
        g_pids (np.array): person IDs for gallery samples.
        g_camids (np.array): camera IDs for gallery samples.
        g_clothes_ids (np.array): clothes IDs for gallery samples.
        unique_g_pids (np.array): unique person IDs for gallery samples.
    """
    if seed is not None:
        set_seed(seed)
        
    result = {}
    cmc, mAP = evaluate(score_mat, q_pids, g_pids, q_camids, g_camids)
    gen_res = pd.Series({'GR_top1': cmc[0], 'GR_top5': cmc[4], 'GR_top10': cmc[9], 'GR_top20': cmc[19], 'GR_mAP': mAP})
    if log:
        logger = logging.getLogger('reid.test')  
        logger.info("----------------- General Results --------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("------------------------------------------------------------")
    if dataset in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']: return cmc[0]

    cmc, mAP = evaluate_with_clothes(score_mat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
    sc_res = pd.Series({'SC_top1': cmc[0], 'SC_top5': cmc[4], 'SC_top10': cmc[9], 'SC_top20': cmc[19], 'SC_mAP': mAP})
    if log:
        logger.info("-------------------- SC Results ----------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("------------------------------------------------------------")
        logger.info("Computing CMC and mAP only for clothes-changing")
    
    cmc, mAP = evaluate_with_clothes(score_mat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    cc_res = pd.Series({'CC_top1': cmc[0], 'CC_top5': cmc[4], 'CC_top10': cmc[9], 'CC_top20': cmc[19], 'CC_mAP': mAP})
    if log:
        logger.info("-------------------- CC Results ----------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("------------------------------------------------------------")
    
    # Ours evaluation metric
    # Compute TAR@FAR
    verification_result = compute_tar_at_far(score_mat, q_pids, g_pids)
    # Compute FNIR@FPIR (for openset evaluation, gallery should be unique)
    opensearch_result = pd.DataFrame()
    for i in range(10): # sample 10 times
        openset_pids = np.sort(np.random.choice(unique_g_pids, int(len(unique_g_pids) * 0.8), replace=False))
        tmp = compute_fnir_at_fpir(merge_score_mat, q_pids, unique_g_pids, openset_pids)
        opensearch_result = pd.concat([opensearch_result, tmp], axis=1)

    opensearch_result_mean = opensearch_result.T.mean(axis=0)
    opensearch_result_std = opensearch_result.T.std(axis=0)[[r'TPIR@0.01%FPIR', r'TPIR@0.10%FPIR', r'TPIR@1.00%FPIR', r'FNIR@0.01%FPIR', r'FNIR@0.10%FPIR', r'FNIR@1.00%FPIR']]
    opensearch_result_std.index = [f'{i}_std' for i in opensearch_result_std.index]
    result = pd.concat([gen_res, sc_res, cc_res, verification_result, opensearch_result_mean, opensearch_result_std], axis=0)
    if log:
        logger.info("--------------- Evaluation Protocol V1.0.0 -----------------")
        logger.info('Number of Probes: {}, Number of Galleries subjects: {}'.format(int(result['num_probe_templates']), int(result['num_gallery_subjects'])))
        # logger.info('Rank1: {:.1%} Rank10: {:.1%} Rank20: {:.1%} mAP: {:.1%}'.format(result['Rank1'], result['Rank10'], result['Rank20'], result['mAP']))
        logger.info('Tar@1.00%FAR:{:.1%} Rank1:{:.1%} FNIR@1%FPIR:{:.1%}±{:.1%}'.format(result[r'TAR@1.00%FAR'], result['GR_top1'], result[r'FNIR@1.00%FPIR'], result[r'FNIR@1.00%FPIR_std']))
        logger.info("------------------------------------------------------------")
    return result

