#ifndef LINHAI_SCORE_HPP
#define LINHAI_SCORE_HPP

#include "types.hpp"
#include <string>
#include <vector>
#include <map>

namespace linhai_score {

struct ContractState {
    int counter = 0;          // 已砸红次数
    int target_count = 3;     // 承包目标数（默认 3）
    bool active = false;      // 是否触发承包
};

struct RedHeadCatch {
    std::vector<int> caught_tiles;   // 已抓到的红头牌（38-array hai code，1..37）
    int per_player_count = 2;        // 每人发的红头数（2/4/6）
};

enum class HuType {
    Ordinary = 0,
    HunYise = 1,
    QingYise = 2,
    ZiYise = 3,
};

struct ChongInputs {
    Hai_Array tehai{};                 // 胡牌时手牌（38 槽）
    Fuuro_Vector melds{};              // 副露
    int white_count_in_hand = 0;       // 手牌白板数
    int white_count_in_melds = 0;      // 副露白板数
    bool is_tsumo = false;             // 自摸
    bool is_qiang_gang = false;        // 抢杠胡
    bool has_tree_active = false;      // 树活（白板暗刻当 3 财神）
    int jikaze = 0;                    // 自风（0=东..3=北）
    int bakaze = 0;                    // 场风
    ContractState contract{};
    RedHeadCatch redhead{};
    int grab_charge_caught_count = 0;  // 本局抓冲数
};

struct ChongBreakdown {
    HuType hu_type = HuType::Ordinary;
    int base_chong = 0;          // 基础冲数 1/2/4/8（应用倍数前）
    int multiplier_2x = 0;       // 硬碰硬 / 特殊牌翻倍叠加层数
    int multiplier_4x = 0;       // 树掉还原 ×4（0 或 1）
    int extra_chong = 0;         // 抓冲（不计入封顶）
    int qianggang_bonus = 0;     // 抢杠胡 +N
    int redhead_bonus = 0;       // 翻屁股加分
    int contract_split = 0;      // 三键承包平摊本数（>0 表示赢家自摸时的两家承包）
    int capped_base = 0;         // 应用倍数和封顶后的基础冲（不含 extra/qianggang/redhead）
    int final_chong = 0;         // 最终冲数 = capped + extra + qianggang + redhead
    int score = 0;               // = final_chong * chong_to_score
    std::string explain;         // 调试可读
};

struct ScoreTable {
    int chong_to_score = 1;
    int base_ordinary = 1;
    int base_qingyise = 8;
    int base_ziyise = 8;
    int base_hunyise_tree = 2;
    int base_hunyise_hard = 4;
    int qianggang_bonus = 2;
    int grab_charge_per_tile = 1;
    int max_base_chong = 8;
    bool extra_chong_uncapped = true;
    std::map<int, int> redhead_tile_to_chong;  // hai 38-code → 冲数
    int redhead_per_player_count_default = 2;
    int contract_target_count_default = 3;

    // === Phase A: 推理侧防守加固（spec §3.5）===
    struct EVRiskWeights {
        float lambda_base = 1.5f;            // spec §3.5.1 默认 1.5
        float lambda_pressure_step = 0.5f;
        float lambda_max = 3.0f;
        float mu_base = 2.0f;
        float mu_pressure_step = 0.5f;
        float mu_max = 5.0f;
        float houjuu_base_coeff = 5500.0f;
        float high_chong_threshold = 4.0f;
    } ev_risk_weights;

    struct BetaoriThresholds {
        float base = 0.15f;
        float meld_drop = 0.03f;
        float qingyise_drop = 0.05f;
        float ev_loss_multiplier = 1.5f;
    } betaori_thresholds;

    struct FeatureWeights {
        float opp_meld_pressure_alpha = 1.0f;
        float opp_qingyise_alarm_alpha = 1.0f;
    } feature_weights;
};

// 从 JSON 文件加载；缺失字段回退到默认值。文件不存在时也返回默认 table。
ScoreTable load_score_table(const std::string& path);

// 主计算函数
ChongBreakdown calc_chong(const ChongInputs& inputs, const ScoreTable& table);

// 牌型识别（公开方便单测）
HuType classify_hu(const Hai_Array& tehai, const Fuuro_Vector& fuuro);

} // namespace linhai_score

#endif
