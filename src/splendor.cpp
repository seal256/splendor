#include <sstream>
#include <cassert>
#include <cctype>
#include <random>

#include "splendor.h"
#include "random_util.h"

namespace splendor {

const int GOLD_GEM = 5; // Index of the gold gem
const std::array<int, NUM_GEMS> GEMS = {0, 1, 2, 3, 4, 5};

const std::vector<char> GEM_STR = {'r', 'g', 'b', 'w', 'k', 'y'}; // red, green, blue, white, black, yellow(gold)
const std::unordered_map<char, int> GEM_STR_TO_VAL = {{'r', 0}, {'g', 1}, {'b', 2}, {'w', 3}, {'k', 4}, {'y', 5}};

const std::vector<std::vector<std::string>> CARDS_STR = {
    {"[k0|r1g1b1w1]","[k0|r1g1b2w1]","[k0|r1b2w2]","[k0|r3g1k1]","[k0|r1g2]","[k0|g2w2]","[k0|g3]","[k1|b4]","[b0|r1g1w1k1]","[b0|r2g1w1k1]","[b0|r2g2w1]","[b0|r1g3b1]","[b0|w1k2]","[b0|g2k2]","[b0|k3]","[b1|r4]","[w0|r1g1b1k1]","[w0|r1g2b1k1]","[w0|g2b2k1]","[w0|b1w3k1]","[w0|r2k1]","[w0|b2k2]","[w0|b3]","[w1|g4]","[g0|r1b1w1k1]","[g0|r1b1w1k2]","[g0|r2b1k2]","[g0|g1b3w1]","[g0|b1w2]","[g0|r2b2]","[g0|r3]","[g1|k4]","[r0|g1b1w1k1]","[r0|g1b1w2k1]","[r0|g1w2k2]","[r0|r1w1k3]","[r0|g1b2]","[r0|r2w2]","[r0|w3]","[r1|w4]"},
    {"[k1|g2b2w3]","[k1|g3w3k2]","[k2|r2g4b1]","[k2|r3g5]","[k2|w5]","[k3|k6]","[b1|r3g2b2]","[b1|g3b2k3]","[b2|b3w5]","[b2|r1w2k4]","[b2|b5]","[b3|b6]","[w1|r2g3k2]","[w1|r3b3w2]","[w2|r4g1k2]","[w2|r5k3]","[w2|r5]","[w3|w6]","[g1|r3g2w3]","[g1|b3w2k2]","[g2|b2w4k1]","[g2|g3b5]","[g2|g5]","[g3|g6]","[r1|r2w2k3]","[r1|r2b3k3]","[r2|g2b4w1]","[r2|w3k5]","[r2|k5]","[r3|r6]"},
    {"[k3|r3g5b3w3]","[k4|r7]","[k4|r6g3k3]","[k5|r7k3]","[b3|r3g3w3k5]","[b4|w7]","[b4|b3w6k3]","[b5|b3w7]","[w3|r5g3b3k3]","[w4|k7]","[w4|r3w3k6]","[w5|w3k7]","[g3|r3b3w5k3]","[g4|b7]","[g4|g3b6w3]","[g5|g3b7]","[r3|g3b5w3k3]","[r4|g7]","[r4|r3g6b3]","[r5|r3g7]"}
};
const std::vector<std::string> NOBLES_STR = {"[3|r4g4]","[3|g4b4]","[3|b4w4]","[3|w4k4]","[3|k4r4]","[3|r3g3b3]","[3|b3g3w3]","[3|b3w3k3]","[3|w3k3r3]","[3|k3r3g3]"};
const std::vector<std::string> ACTIONS_STR = {"s", // skip move
    "tr2","tg2","tb2","tw2","tk2","tr1g1b1","tr1g1w1","tr1g1k1","tr1b1w1","tr1b1k1","tr1w1k1","tg1b1w1","tg1b1k1","tg1w1k1","tb1w1k1", // take 
    "r0n0","r0n1","r0n2","r0n3","r1n0","r1n1","r1n2","r1n3","r2n0","r2n1","r2n2","r2n3", // reserve
    "p0n0","p0n1","p0n2","p0n3","p1n0","p1n1","p1n2","p1n3","p2n0","p2n1","p2n2","p2n3", // purchase
    "h0","h1","h2", // purchase from hand
    "c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15","c16","c17","c18","c19","c20","c21","c22","c23","c24","c25","c26","c27","c28","c29","c30","c31","c32","c33","c34","c35","c36","c37","c38","c39" // new card (choice node actions)
};


std::ostream& operator<<(std::ostream& os, const GemSet& gem_set) {
    for (int gem = 0; gem < NUM_GEMS; ++gem) {
        if (gem_set.gems[gem] > 0) {
            os << GEM_STR[gem] << gem_set.gems[gem];
        }
    }
    return os;
}

GemSet GemSet::from_str(const std::string& input) {
    // reads some number of char + int pairs, e.g. r2g1b4
    GemSet gem_set;
    int pos = 0;
    while (pos < input.size()) {
        // Check if the current character is a gem type
        char gem_char = input[pos];
        if (GEM_STR_TO_VAL.find(gem_char) != GEM_STR_TO_VAL.end()) {
            int gem = GEM_STR_TO_VAL.at(gem_char);
            pos++;

            // Check if the next character is a digit (count)
            int count = 1; // Default count is 1 if no number is specified
            if (pos < input.size() && isdigit(input[pos])) {
                count = input[pos] - '0'; // reads at most 1 digit
                pos++;
            }

            gem_set.gems[gem] += count;
        } else {
            throw std::invalid_argument("GemSet input string is invalid");
            // pos++; // ignore errors
        }
    }
    return gem_set;
}

int GemSet::sum() const {
    int n = 0;
    for (const int & count : gems) {
        n += count;
    }
    return n;
}

int GemSet::unique() const {
    int n = 0;
    for (const int & count : gems) {
        if (count > 0) 
            n++;
    }
    return n;
}

Noble Noble::from_str(const std::string& input) {
    assert(input[0] == '[' && input[2] == '|' && input[input.size()-1] == ']');
    int points = input[1] - '0'; // only one digit is expected
    GemSet price = GemSet::from_str(input.substr(3, input.size()-4));
    return Noble(points, price);
}

std::ostream& operator<<(std::ostream& os, const Noble& noble) {
    os << "[" << noble.points << "|" << noble.price << "]";
    return os;
}

std::string Noble::to_str() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

Card Card::from_str(const std::string& input) {
    assert(input[0] == '[' && input[3] == '|' && input[input.size() - 1] == ']');
    int gem = GEM_STR_TO_VAL.at(input[1]);
    int points = input[2] - '0'; // only one digit is expected
    GemSet price = GemSet::from_str(input.substr(4, input.size() - 5));
    return Card(gem, points, price);
}

std::ostream& operator<<(std::ostream& os, const Card& card) {
    os << "[" << GEM_STR[card.gem] << card.points << "|" << card.price << "]";
    return os;
}

std::string Card::to_str() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::vector<Noble> init_nobles() {
    std::vector<Noble> nobles;
    for (const std::string& noble_str : NOBLES_STR) {
        nobles.push_back(Noble::from_str(noble_str));
    }
    return nobles;
}

std::vector<std::vector<Card>> init_cards() {
    std::vector<std::vector<Card>> cards;
    cards.resize(CARD_LEVELS);
    for (int level = 0; level < CARD_LEVELS; level ++) {
        for (const std::string& card_str : CARDS_STR[level]) {
            cards[level].push_back(Card::from_str(card_str));
        }
    }
    return cards;
}

const std::vector<Noble> NOBLES = init_nobles();
const std::vector<std::vector<Card>> CARDS = init_cards();

std::ostream& operator<<(std::ostream& os, const SplendorPlayerState& player) {
    os << "player " << player.id << " | points " << player.points << "\n"
       << "card gems: " << player.card_gems << "\n"
       << "gems: " << player.gems << "\n"
       << "hand: ";
    for (const Card* card : player.hand_cards) {
        os << *card << " ";
    }
    os << "\n";
    return os;
}

const std::unordered_map<char, ActionType> ACTION_STR_TO_VAL = {
    {'s', ActionType::SKIP},
    {'t', ActionType::TAKE},
    {'r', ActionType::RESERVE},
    {'p', ActionType::PURCHASE},
    {'h', ActionType::PURCHASE_HAND},
    {'c', ActionType::NEW_TABLE_CARD}
};

const std::unordered_map<ActionType, char> ACTION_VAL_TO_STR = {
    {ActionType::SKIP, 's'},
    {ActionType::TAKE, 't'},
    {ActionType::RESERVE, 'r'},
    {ActionType::PURCHASE, 'p'},
    {ActionType::PURCHASE_HAND, 'h'},
    {ActionType::NEW_TABLE_CARD, 'c'}
};

Action Action::from_str(const std::string& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input string is empty");
    }

    ActionType action_type;
    GemSet gems;
    int level = -1;
    int pos = -1;

    auto action_type_it = ACTION_STR_TO_VAL.find(input[0]);
    if (action_type_it == ACTION_STR_TO_VAL.end())
        throw std::invalid_argument("Invalid action type in action " + input);
    action_type = action_type_it->second;
    
    if (action_type == ActionType::TAKE) {
        gems = GemSet::from_str(input.substr(1));

    } else if (action_type == ActionType::RESERVE || action_type == ActionType::PURCHASE) {
        size_t separator_pos = input.find('n');
        if (separator_pos == std::string::npos || separator_pos == 0 || separator_pos == input.size() - 1) {
            throw std::invalid_argument("Invalid format for level and position in action " + input);
        }
        level = std::stoi(input.substr(1, separator_pos - 1));
        pos = std::stoi(input.substr(separator_pos + 1));
    
    } else if (action_type == ActionType::PURCHASE_HAND || action_type == ActionType::NEW_TABLE_CARD) {
        pos = std::stoi(input.substr(1));
    }

    return Action(action_type, gems, level, pos);
}

std::ostream& operator<<(std::ostream& os, const Action& action) {
    auto action_str_it = ACTION_VAL_TO_STR.find(action.type);
    if (action_str_it == ACTION_VAL_TO_STR.end()) {
        throw std::invalid_argument("Unknown action type");
    }
    os << action_str_it->second;
    
    switch (action.type) {
        case ActionType::TAKE:
            os << action.gems;
            break;

        case ActionType::RESERVE:
        case ActionType::PURCHASE:
            os << action.level << "n" << action.pos;
            break;

        case ActionType::PURCHASE_HAND:
        case ActionType::NEW_TABLE_CARD:
            os << action.pos;
            break;

        case ActionType::SKIP:
            break;

        default:
            throw std::invalid_argument("Unknown action type");
    }
    return os;    
}

std::string Action::to_str() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::vector<Action> init_actions() {
    std::vector<Action> actions;
    for (const auto& action_str : ACTIONS_STR) {
        actions.push_back(Action::from_str(action_str));
    }
    return actions;
}
const std::vector<Action> ACTIONS = init_actions();

std::unordered_map<ActionType, std::vector<int>> init_action_type_ids(const std::vector<Action>& actions) {
    std::unordered_map<ActionType, std::vector<int>> ids;
    for (int n = 0; n < actions.size(); ++n) {
        const auto& action = actions[n];
        ids[action.type].push_back(n);
    }
    return ids;
}

std::unordered_map<ActionType, std::vector<int>> ACTION_TYPE_IDS = init_action_type_ids(ACTIONS); // indices for all action types

const std::unordered_map<int, SplendorGameRules> DEFAULT_RULES = {{2, SplendorGameRules(2)}, {3, SplendorGameRules(3)}, {4, SplendorGameRules(4)}};

SplendorGameState::SplendorGameState(int num_players, const SplendorGameRules* rules)
    : rules(rules ? rules : &DEFAULT_RULES.at(num_players)),
      round(0),
      player_to_move(0),
      skips(0),
      table_card_needed(false),
      deck_level(0) 
    {

    // Initialize nobles
    nobles.reserve(NOBLES.size());
    for (const Noble& noble : NOBLES) {
        nobles.push_back(&noble);
    }
    random_shuffle(nobles.begin(), nobles.end());
    nobles.resize(this->rules->max_nobles);

    // Initialize decks and cards
    decks.resize(CARD_LEVELS);
    cards.resize(CARD_LEVELS);
    for (int level = 0; level < CARD_LEVELS; ++level) {
        auto & deck_cards = decks[level];
        deck_cards.reserve(CARDS[level].size());
        for (const Card& card : CARDS[level]) {
            deck_cards.push_back(&card);
        }
        random_shuffle(deck_cards.begin(), deck_cards.end());
        int open_cards = this->rules->max_open_cards;
        cards[level].resize(open_cards);
        std::copy(deck_cards.end() - open_cards, deck_cards.end(), cards[level].begin()); // copy from the end of the shuffled array
        deck_cards.resize(CARDS[level].size() - open_cards);
    }

    // Initialize gems
    gems = GemSet();
    for (int gem = 0; gem < NUM_GEMS - 1; ++gem) {
        gems.gems[gem] = this->rules->max_gems;
    }
    gems.gems[GOLD_GEM] = this->rules->max_gold;

    // Initialize players
    players.reserve(num_players);
    for (int id = 0; id < num_players; ++id) {
        players.emplace_back(SplendorPlayerState{id});
    }
}

// std::vector<int> SplendorGameState::get_actions() const {
//     std::vector<int> actions;
//     actions.reserve(ACTIONS.size() / 2);
//     for (int action = 0; action < ACTIONS.size(); action++) {
//         auto [valid, err] = verify_action(action);
//         if (valid) {
//             actions.push_back(action);
//         }
//     }
//     return actions;
// }

// Faster but more dangerous version, that assumes specific order of the ACTIONS list. Will likely break if the game rules are changed
std::vector<int> SplendorGameState::get_actions() const {
    std::vector<int> actions;
    actions.reserve(ACTIONS.size() / 2);

    if (table_card_needed) {
        // Move of the gods of randomness, no player is involved
        const auto& action_ids = ACTION_TYPE_IDS[ActionType::NEW_TABLE_CARD];
        actions.insert(actions.end(), action_ids.begin(), action_ids.begin() + decks[deck_level].size());
        return actions;
    }

    const SplendorPlayerState& player = players[player_to_move];

    // 1. Take new gems
    // Two same gems
    int player_gems_sum = player.gems.sum();
    if (player_gems_sum < rules->max_player_gems - rules->max_same_gems_take) {
        const auto& action_ids = ACTION_TYPE_IDS[ActionType::TAKE];
        for (int gem = 0; gem < NUM_GEMS - 1; ++gem) { // Exclude gold gem
            if (gems.gems[gem] >= rules->min_same_gems_stack) {
                actions.push_back(action_ids[gem]); // be careful to arrange take actions so that take same go first
            }
        }
    }

    // Three distinct gems
    if (player_gems_sum <= rules->max_player_gems - rules->max_gems_take) {
        const auto& action_ids = ACTION_TYPE_IDS[ActionType::TAKE];
        for (int n = NUM_GEMS - 1; n < action_ids.size(); ++n) {
            const auto& action = ACTIONS[action_ids[n]];
            bool can_take = true;
            for (int gem = 0; gem < NUM_GEMS - 1; ++gem) {
                if (action.gems.gems[gem] > gems.gems[gem]) {
                    can_take = false;
                    break;
                }
            }
            if (can_take) {
                actions.push_back(action_ids[n]);
            }
        }
    }

    // 2. Reserve a card
    if (player.hand_cards.size() < rules->max_hand_cards) {
        for (const auto& action_id : ACTION_TYPE_IDS[ActionType::RESERVE]) {
            const auto& action = ACTIONS[action_id];
            if (action.pos < cards[action.level].size()) {
                actions.push_back(action_id);
            }
        }
    }

    // 3. Purchase a card from the table
    for (int level = 0; level < cards.size(); ++level) {
        for (int pos = 0; pos < cards[level].size(); ++pos) {
            if (player_can_afford_card(player, *cards[level][pos])) {
                actions.push_back(ACTION_TYPE_IDS[ActionType::PURCHASE][level * rules->max_open_cards + pos]);
            }
        }
    }

    // 4. Purchase a card from the hand
    if (!player.hand_cards.empty()) {
        for (int pos = 0; pos < player.hand_cards.size(); ++pos) {
            if (player_can_afford_card(player, *player.hand_cards[pos])) {
                actions.push_back(ACTION_TYPE_IDS[ActionType::PURCHASE_HAND][pos]);
            }
        }
    }

    // 0. Skip the move
    if (actions.empty()) {
        actions.push_back(ACTION_TYPE_IDS[ActionType::SKIP][0]);
    }

    return actions;
}


int SplendorGameState::active_player() const {
    if (table_card_needed) {
        return CHANCE_PLAYER;
    }
    return player_to_move;
}

const std::unordered_map<ActionError, std::string> ACTION_ERROR_STRINGS = {
    {ActionError::NONE, "All good"},
    {ActionError::INVALID_ACTION_ID, "Invalid action id"},
    {ActionError::CANNOT_TAKE_MORE_THAN_MAX_GEMS, "Can't take more than maximum allowed gems"},
    {ActionError::NOT_ENOUGH_GEMS_IN_STACK, "Not enough gems in stack"},
    {ActionError::CANNOT_TAKE_MORE_THAN_MAX_IDENTICAL_GEMS, "Can't take more than maximum identical gems"},
    {ActionError::MUST_TAKE_ALL_IDENTICAL_OR_ALL_DIFFERENT, "Must take all identical or all different gems"},
    {ActionError::PLAYER_CANNOT_HAVE_MORE_GEMS, "Player can't have more gems"},
    {ActionError::CANNOT_TAKE_GOLD_GEM, "Can't take gold gem"},
    {ActionError::NOT_ENOUGH_GEMS_ON_TABLE, "Not enough gems on table"},
    {ActionError::INVALID_DECK_LEVEL, "Invalid deck level"},
    {ActionError::INVALID_CARD_POSITION, "Invalid card position"},
    {ActionError::PLAYER_CANNOT_RESERVE_MORE_CARDS, "Player can't reserve more cards"},
    {ActionError::PLAYER_CANNOT_AFFORD_CARD, "Player can't afford the card"},
    {ActionError::GAME_REQUIRES_NEW_TABLE_CARD, "Game requires new table card"},
    {ActionError::GAME_DOES_NOT_NEED_NEW_TABLE_CARD, "Game doesn't need new table card"},
    {ActionError::INVALID_ACTION_TYPE, "Invalid action type"}
};

std::string action_error_to_string(ActionError err) {
    auto it = ACTION_ERROR_STRINGS.find(err);
    return it != ACTION_ERROR_STRINGS.end() ? it->second : "Unknown error";
}

std::pair<bool, ActionError> SplendorGameState::verify_action(const int action_id) const {
    if (action_id < 0 || action_id >= ACTIONS.size()) {
        return {false, ActionError::INVALID_ACTION_ID};
    }

    const Action& action = ACTIONS[action_id];

    if (table_card_needed) {
        if (action.type != ActionType::NEW_TABLE_CARD) {
            return {false, ActionError::GAME_REQUIRES_NEW_TABLE_CARD};
        }
        if (deck_level < 0 || deck_level >= decks.size()) {
            return {false, ActionError::INVALID_DECK_LEVEL};
        }
        if (action.pos < 0 || action.pos >= decks[deck_level].size()) {
            return {false, ActionError::INVALID_CARD_POSITION};
        }
        return {true, ActionError::NONE};

    } else {
        if (action.type == ActionType::NEW_TABLE_CARD) {
            return {false, ActionError::GAME_DOES_NOT_NEED_NEW_TABLE_CARD};
        }
    }
    
    if (action.type == ActionType::SKIP) {
        return {true, ActionError::NONE};
    }

    const SplendorPlayerState& player = players[player_to_move];

    if (action.type == ActionType::TAKE) {
        int unique_gems_take = action.gems.unique();
        int total_gems_take = action.gems.sum();

        if (total_gems_take > rules->max_gems_take) {
            return {false, ActionError::CANNOT_TAKE_MORE_THAN_MAX_GEMS};
        }
        if (unique_gems_take == 1) {
            auto gem_iter = std::find_if(action.gems.gems.begin(), action.gems.gems.end(), [](int x) {return x > 0;});
            int color = gem_iter - action.gems.gems.begin();
            if (this->gems.gems[color] < rules->min_same_gems_stack) {
                return {false, ActionError::NOT_ENOUGH_GEMS_IN_STACK};
            }
            if (*gem_iter > rules->max_same_gems_take) {
                return {false, ActionError::CANNOT_TAKE_MORE_THAN_MAX_IDENTICAL_GEMS};
            }
        }
        if (unique_gems_take > 1 && unique_gems_take != total_gems_take) {
            return {false, ActionError::MUST_TAKE_ALL_IDENTICAL_OR_ALL_DIFFERENT};
        }
        if (player.gems.sum() + total_gems_take > rules->max_player_gems) {
            return {false, ActionError::PLAYER_CANNOT_HAVE_MORE_GEMS};
        }

        for (int gem = 0; gem < NUM_GEMS; ++gem) {
            int gem_count = action.gems.gems[gem];
            if (gem_count > 0) {
                if (gem == GOLD_GEM) {
                    return {false, ActionError::CANNOT_TAKE_GOLD_GEM};
                }
                if (gem_count > this->gems.gems[gem]) {
                    return {false, ActionError::NOT_ENOUGH_GEMS_ON_TABLE};
                }
            }
        }
        return {true, ActionError::NONE};
    }

    if (action.type == ActionType::RESERVE) {
        if (action.level < 0 || action.level >= cards.size()) {
            return {false, ActionError::INVALID_DECK_LEVEL};
        }
        if (action.pos < 0 || action.pos >= cards[action.level].size()) {
            return {false, ActionError::INVALID_CARD_POSITION};
        }
        if (player.hand_cards.size() >= rules->max_hand_cards) {
            return {false, ActionError::PLAYER_CANNOT_RESERVE_MORE_CARDS};
        }
        return {true, ActionError::NONE};
    }

    if (action.type == ActionType::PURCHASE) {
        if (action.level < 0 || action.level >= cards.size()) {
            return {false, ActionError::INVALID_DECK_LEVEL};
        }
        if (action.pos < 0 || action.pos >= cards[action.level].size()) {
            return {false, ActionError::INVALID_CARD_POSITION};
        }
        const Card* card = cards[action.level][action.pos];
        if (!player_can_afford_card(player, *card)) {
            return {false, ActionError::PLAYER_CANNOT_AFFORD_CARD};
        }
        return {true, ActionError::NONE};
    }

    if (action.type == ActionType::PURCHASE_HAND) {
        if (action.pos < 0 || action.pos >= player.hand_cards.size()) {
            return {false, ActionError::INVALID_CARD_POSITION};
        }
        const Card* card = player.hand_cards[action.pos];
        if (!player_can_afford_card(player, *card)) {
            return {false, ActionError::PLAYER_CANNOT_AFFORD_CARD};
        }
        return {true, ActionError::NONE};
    }

    return {false, ActionError::INVALID_ACTION_TYPE};
}

void SplendorGameState::apply_action(const int action_id) {
    // Uncomment for more safety if needed
    // auto [is_valid, error] = verify_action(action_id);
    // if (!is_valid) {
    //     throw std::invalid_argument(action_error_to_string(error));
    // }

    const Action& action = ACTIONS[action_id];
    SplendorPlayerState& player = players[player_to_move];

    if (player_to_move == 0) {
        skips = 0;
    }

    switch (action.type) {
        case ActionType::SKIP: {
            skips++;
            increment_player_to_move();
            break;
        }

        case ActionType::TAKE: {
            for (int gem = 0; gem < NUM_GEMS; ++gem) {
                int gem_count = action.gems.gems[gem];
                if (gem_count > 0) {
                    player.gems.gems[gem] += gem_count;
                    this->gems.gems[gem] -= gem_count;
                }
            }
            increment_player_to_move();
            break;
        }

        case ActionType::RESERVE: {
            const Card* card = cards[action.level][action.pos];
            cards[action.level].erase(cards[action.level].begin() + action.pos);
            set_table_card_needed(action.level);

            player.hand_cards.push_back(card);
            if (this->gems.gems[GOLD_GEM] > 0) {
                player.gems.gems[GOLD_GEM]++;
                this->gems.gems[GOLD_GEM]--;
            }

            increment_player_to_move();
            break;
        }

        case ActionType::PURCHASE: {
            const Card* card = cards[action.level][action.pos];
            cards[action.level].erase(cards[action.level].begin() + action.pos);
            purchase_card(player, *card);
            set_table_card_needed(action.level);
            get_noble(player);
            increment_player_to_move();
            break;
        }

        case ActionType::PURCHASE_HAND: {
            const Card* card = player.hand_cards[action.pos];
            player.hand_cards.erase(player.hand_cards.begin() + action.pos);
            purchase_card(player, *card);
            get_noble(player);
            increment_player_to_move();
            break;
        }

        case ActionType::NEW_TABLE_CARD: {
            const Card* new_card = decks[deck_level][action.pos];
            decks[deck_level].erase(decks[deck_level].begin() + action.pos);
            cards[deck_level].push_back(new_card);
            table_card_needed = false;
            break;
        }

        default:
            throw std::invalid_argument(action_error_to_string(ActionError::INVALID_ACTION_TYPE));
    }
}

bool SplendorGameState::is_terminal() const {
    // The game stops if all players skipped the move in the current round
    if (skips >= players.size()) {
        return true;
    }

    // Or one of them got the required number of win points
    if (this->active_player() == 0) { // ensures that all players made equal number of moves
        for (const auto& player : players) {
            if (player.points >= rules->win_points) {
                return true;
            }
        }
    }

    return false;
}

int SplendorGameState::move_num() const {
    return round;
}

std::vector<int> SplendorGameState::_get_winners() const {
    std::vector<int> candidate_ids;
    int max_points = -1;
    for (const auto& player : players) {
        if (player.points > max_points)
            max_points = player.points;
    }

    // no winner
    if (max_points < rules->win_points) {
        return candidate_ids;
    }

    for (const auto& player : players) {
        if (player.points >= max_points) {
            candidate_ids.push_back(player.id);
        }
    }

    // If there's only one candidate, they are the winner
    if (candidate_ids.size() <= 1) {
        return candidate_ids;
    }

    // In case of a tie, the player with the fewest development cards wins
    int min_cards = 1000;
    for (int id : candidate_ids) {
        int num_cards = players[id].card_gems.sum(); // Sum of all card gems represents the number of development cards
        if (num_cards < min_cards) {
            min_cards = num_cards;
        }
    }

    // there still may be a tie
    std::vector<int> winner_ids;
    for (int id : candidate_ids) {
        int num_cards = players[id].card_gems.sum();
        if (num_cards <= min_cards) {
            winner_ids.push_back(id);
        }
    }

    return winner_ids;
}

std::vector<double> SplendorGameState::rewards() const {
    std::vector<double> rewards(players.size(), 0.0);
    std::vector<int> winner_ids = _get_winners();
    for (int id : winner_ids) {
        rewards[id] = 1.0 / winner_ids.size();
    }

    return rewards;
}

std::shared_ptr<GameState> SplendorGameState::clone() const {
    return std::make_shared<SplendorGameState>(*this); // Uses the default copy constructor. Shallow copy of the arrays of pointers is intentional
}

void SplendorGameState::print(std::ostream& os) const {
    os << *this;
}

void SplendorGameState::to_json(nlohmann::json& j) const {
    splendor::to_json(j, *this);
}

void SplendorGameState::increment_player_to_move() {
    player_to_move = (player_to_move + 1) % rules->num_players;
    if (player_to_move == 0) {
        round++;
    }
}

void SplendorGameState::set_table_card_needed(int level) {
    if (!decks[level].empty()) {
        table_card_needed = true;
        deck_level = level;
    }
}

bool SplendorGameState::player_can_afford_card(const SplendorPlayerState& player, const Card& card) const {
    int gold_to_pay = 0;
    for (int gem = 0; gem < NUM_GEMS; ++gem) {
        int gem_to_pay = card.price.gems[gem] - player.card_gems.gems[gem];
        if (gem_to_pay > 0) {
            int gem_available = player.gems.gems[gem];
            if (gem_to_pay > gem_available) {
                gold_to_pay += gem_to_pay - gem_available;
            }
        }
    }
    return gold_to_pay <= player.gems.gems[GOLD_GEM];
}

void SplendorGameState::purchase_card(SplendorPlayerState& player, const Card& card) {
    int gold_to_pay = 0;
    for (int gem = 0; gem < NUM_GEMS; ++gem) {
        int gem_to_pay = card.price.gems[gem] - player.card_gems.gems[gem];
        if (gem_to_pay > 0) {
            int gem_available = player.gems.gems[gem];
            if (gem_to_pay > gem_available) {
                gold_to_pay += gem_to_pay - gem_available;
                this->gems.gems[gem] += gem_available;
                player.gems.gems[gem] = 0;
            } else {
                this->gems.gems[gem] += gem_to_pay;
                player.gems.gems[gem] = gem_available - gem_to_pay;
            }
        }
    }

    player.gems.gems[GOLD_GEM] -= gold_to_pay;
    this->gems.gems[GOLD_GEM] += gold_to_pay;

    player.card_gems.gems[card.gem]++;
    player.points += card.points;
}

void SplendorGameState::get_noble(SplendorPlayerState& player) {
    int noble_idx = -1;
    for (size_t n = 0; n < nobles.size(); ++n) {
        bool can_afford = true;
        for (int gem = 0; gem < NUM_GEMS; ++gem) {
            if (player.card_gems.gems[gem] < nobles[n]->price.gems[gem]) {
                can_afford = false;
                break;
            }
        }
        if (can_afford) {
            noble_idx = n;
            break; // We take the first appropriate noble for simplicity. There's no action to select the noble if the player can choose among more than one
        }
    }

    if (noble_idx >= 0) {
        const Noble* noble = nobles[noble_idx];
        nobles.erase(nobles.begin() + noble_idx);
        player.points += noble->points;
    }
}

std::ostream& operator<<(std::ostream& os, const SplendorGameState& state) {
    os << "round: " << state.round << " player to move: " << state.active_player() << "\n";

    os << "nobles: ";
    for (const Noble* noble : state.nobles) {
        os << *noble << " ";
    }
    os << "\n";

    for (size_t n = 0; n < state.cards.size(); ++n) {
        os << (state.cards.size() - n) << ": ";
        for (const Card* card : state.cards[n]) {
            if (card) {
                os << *card << " ";
            }
        }
        os << "\n";
    }

    os << "gems: " << state.gems << "\n";

    for (const SplendorPlayerState& player : state.players) {
        os << player;
    }

    return os;
}

void to_json(nlohmann::json& j, const GemSet& gem_set) {
    j = gem_set.gems;
}

void to_json(nlohmann::json& j, const SplendorPlayerState& player_state) {
    std::vector<std::string> hand_cards;
    for (const auto card: player_state.hand_cards)
        hand_cards.push_back(card->to_str());

    j = nlohmann::json{
        {"id", player_state.id}
    };
    
    if (!player_state.card_gems.gems.empty()) {
        j["card_gems"] = player_state.card_gems;
    }
    if (!player_state.gems.gems.empty()) {
        j["gems"] = player_state.gems;
    }
    if (!hand_cards.empty()) {
        j["hand_cards"] = hand_cards;
    }
    if (player_state.points) {
        j["points"] = player_state.points;
    }
}

void to_json(nlohmann::json& j, const SplendorGameRules& rules) {
    j = nlohmann::json{
        {"num_players", rules.num_players},
        {"max_open_cards", rules.max_open_cards},
        {"max_hand_cards", rules.max_hand_cards},
        {"win_points", rules.win_points},
        {"max_player_gems", rules.max_player_gems},
        {"max_nobles", rules.max_nobles},
        {"max_gems_take", rules.max_gems_take},
        {"max_same_gems_take", rules.max_same_gems_take},
        {"min_same_gems_stack", rules.min_same_gems_stack},
        {"max_gold", rules.max_gold},
        {"max_gems", rules.max_gems}
    };
}

void to_json(nlohmann::json& j, const SplendorGameState& state) {
    std::vector<std::string> nobles;
    for (const auto noble : state.nobles)
        nobles.push_back(noble->to_str());

    std::vector<std::vector<std::string>> decks;
    for (const auto& deck : state.decks) {
        std::vector<std::string> deck_strs;
        for (const auto card : deck)
            deck_strs.push_back(card->to_str());
        decks.push_back(deck_strs);
    }

    std::vector<std::vector<std::string>> cards;
    for (const auto& card_row : state.cards) {
        std::vector<std::string> card_row_strs;
        for (const auto card : card_row)
            card_row_strs.push_back(card->to_str());
        cards.push_back(card_row_strs);
    }

    j = nlohmann::json{
        // {"rules", state.rules},
        {"nobles", nobles},
        {"decks", decks},
        {"cards", cards},
        {"gems", state.gems},
        {"players", state.players}
    };
    
    if (state.round) {
        j["round"] = state.round;
    }
    if (state.player_to_move) {
        j["player_to_move"] = state.player_to_move;
    }
    if (state.skips) {
        j["skips"] = state.skips;
    }
    if (state.table_card_needed) {
        j["table_card_needed"] = state.table_card_needed;
    }
    if (state.deck_level) {
        j["deck_level"] = state.deck_level;
    }
}


} // namespace splendor