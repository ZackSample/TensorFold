//! Whom a question asks about: the user (I, my), the model (you, your) or someone the user knows.
const std = @import("std");
const units = @import("slide_units.zig");
const says = units.says;
const delimiters = units.delimiters;
const Allocator = std.mem.Allocator;

/// Whether a question asks yes or no of the user's own life (Is it blue? Do I like it?), not "Do you remember...?".
pub fn yesNo(question: []const u8) bool {
    const verbs = [_][]const u8{ "is", "are", "am", "was", "were", "do", "does", "did", "have", "has", "had", "can", "could", "will", "would", "should" };
    var it = std.mem.tokenizeAny(u8, question, delimiters);
    const first = it.next() orelse return false;
    const second = it.next() orelse return false;
    for (verbs) |v| if (std.ascii.eqlIgnoreCase(first, v)) return !std.ascii.eqlIgnoreCase(second, "you");
    return false;
}

/// Whether a question is still the user's own: it says I or my, never your, and you only once ("Do you remember").
pub fn firstPerson(question: []const u8) bool {
    var mine = false;
    var you: usize = 0;
    var it = std.mem.tokenizeAny(u8, question, delimiters);
    while (it.next()) |w| {
        for ([_][]const u8{ "i", "my", "me", "mine", "our", "we", "us", "i'm", "i've" }) |f| mine = mine or std.ascii.eqlIgnoreCase(w, f);
        for ([_][]const u8{ "your", "yours", "yourself", "you're", "you've" }) |f| if (std.ascii.eqlIgnoreCase(w, f)) return false;
        you += @intFromBool(std.ascii.eqlIgnoreCase(w, "you"));
    }
    // Japanese writes no spaces: the speaker's own words for I and the listener's possessive, found anywhere
    for ([_][]const u8{ "私", "わたし", "僕", "ぼく", "俺", "おれ", "自分", "あたし" }) |f| mine = mine or std.mem.indexOf(u8, question, f) != null;
    for ([_][]const u8{ "あなたの", "君の", "きみの" }) |f| if (std.mem.indexOf(u8, question, f) != null) return false;
    return mine and you <= 1;
}

/// Whom a told fact speaks of: the user (or someone or something they know), or the model itself ("Your name is Nyx.").
pub const About = enum { user, self };

/// The user's own words: a question that says one of them asks about the user or their things.
const users_words = [_][]const u8{ "my", "mine", "me", "our", "ours", "us", "myself", "i'm", "i've", "am" };

/// The model's own words when the user speaks to it (you, your).
const models_words = [_][]const u8{ "your", "yours", "yourself", "you're", "you've", "you'll", "you'd" };

/// Verbs a question asks with, after which "you" is the one asked ("Do you know...?"), not the one asked about.
const auxiliaries = [_][]const u8{ "do", "does", "did", "can", "could", "will", "would", "should", "are", "were", "have", "has", "had", "may", "might", "shall", "must" };

fn isAny(w: []const u8, list: []const []const u8) bool {
    for (list) |x| if (std.ascii.eqlIgnoreCase(w, x)) return true;
    return false;
}

/// Whether a told fact speaks to the model (you, your): the wording cue before the model judges it a fact about itself.
pub fn addressesModel(fact: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, fact, delimiters);
    while (it.next()) |w| if (std.ascii.eqlIgnoreCase(w, "you") or isAny(w, &models_words)) return true;
    for ([_][]const u8{ "あなた", "君", "きみ" }) |f| if (std.mem.indexOf(u8, fact, f) != null) return true;
    return false;
}

/// Whether a question asks the model about itself: it says your, or you as the one it is about ("Who made you?",
/// "What should I call you?"), and nothing of the user's own (my, me, am: "Do you know who I am?" is the user's).
pub fn secondPerson(question: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, question, delimiters);
    var yours = false; // your, yourself, you're
    var you = false; // you, anywhere
    var done_to = false; // you after a verb that is not the asking one ("call you"), not "Do you"
    var subject = false; // I or we: the user asking
    var last: []const u8 = "";
    while (it.next()) |w| {
        if (isAny(w, &users_words)) return false;
        yours = yours or isAny(w, &models_words);
        subject = subject or std.ascii.eqlIgnoreCase(w, "i") or std.ascii.eqlIgnoreCase(w, "we");
        if (std.ascii.eqlIgnoreCase(w, "you")) {
            you = true;
            done_to = done_to or !isAny(last, &auxiliaries);
        }
        last = w;
    }
    for ([_][]const u8{ "私", "わたし", "僕", "ぼく", "俺", "おれ", "自分", "あたし" }) |f| if (std.mem.indexOf(u8, question, f) != null) return false;
    for ([_][]const u8{ "あなた", "君", "きみ" }) |f| if (std.mem.indexOf(u8, question, f) != null) return true;
    // with the user as subject ("Do you know what I like?") only a you something is done to asks about the model
    return yours or (you and (!subject or done_to));
}

/// The question asked about the user instead of the model (your as my, you as I or me), or null when it says neither.
pub fn asUser(a: Allocator, question: []const u8) !?[]const u8 {
    const swaps = [_][2][]const u8{ .{ "your", "my" }, .{ "yours", "mine" }, .{ "yourself", "myself" }, .{ "you're", "I'm" }, .{ "you've", "I've" }, .{ "you'll", "I'll" }, .{ "you'd", "I'd" } };
    // "you" after these is the one something is done to or for: me, else the one doing it: I
    const object_before = [_][]const u8{ "to", "for", "with", "about", "of", "at", "from", "by", "than", "like", "call", "called", "name", "named", "made", "make", "built", "created", "teach", "tell", "told", "give", "gave", "run", "runs", "serve", "serves", "describe", "know", "love", "see", "hear", "help", "ask", "trained", "train" };
    // the question's words and what lies between them, the words swapped in place
    var parts: std.ArrayList([]const u8) = .empty;
    var swapped = false;
    var at: usize = 0;
    var prev: []const u8 = "";
    while (at < question.len) {
        var end = at;
        while (end < question.len and (std.ascii.isAlphabetic(question[end]) or question[end] == '\'')) end += 1;
        if (end == at) {
            while (end < question.len and !(std.ascii.isAlphabetic(question[end]) or question[end] == '\'')) end += 1;
            try parts.append(a, question[at..end]);
            at = end;
            continue;
        }
        const word = question[at..end];
        var to: ?[]const u8 = for (swaps) |s| {
            if (std.ascii.eqlIgnoreCase(word, s[0])) break s[1];
        } else null;
        if (to == null and std.ascii.eqlIgnoreCase(word, "you")) to = if (!isAny(prev, &auxiliaries) and isAny(prev, &object_before)) "me" else "I";
        if (to) |t| {
            swapped = true;
            const upper = t[0] == 'I' or (at == 0 and std.ascii.isUpper(word[0]));
            try parts.append(a, if (upper) try std.fmt.allocPrint(a, "{c}{s}", .{ std.ascii.toUpper(t[0]), t[1..] }) else t);
        } else try parts.append(a, word);
        prev = word;
        at = end;
    }
    if (!swapped) return null;
    // the verb follows its new subject: "are you" asked of the user is "am I", "you are" is "I am"
    for (parts.items, 0..) |p, i| {
        if (!std.mem.eql(u8, p, "I")) continue;
        for ([_]?usize{ if (i >= 2) i - 2 else null, if (i + 2 < parts.items.len) i + 2 else null }) |near| {
            const j = near orelse continue;
            const verb = parts.items[j];
            const turned: ?[]const u8 = if (std.ascii.eqlIgnoreCase(verb, "are")) "am" else if (std.ascii.eqlIgnoreCase(verb, "were")) "was" else null;
            if (turned) |t| parts.items[j] = if (std.ascii.isUpper(verb[0])) try std.fmt.allocPrint(a, "{c}{s}", .{ std.ascii.toUpper(t[0]), t[1..] }) else t;
        }
    }
    return try std.mem.concat(a, u8, parts.items);
}

/// Whether a question is about the model itself (you, your): no fact the user tells answers it.
pub fn toModel(question: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, question, delimiters);
    while (it.next()) |w| if (std.ascii.eqlIgnoreCase(w, "you") or std.ascii.eqlIgnoreCase(w, "your")) return true;
    return false;
}

/// The question asked of the model instead (my as your, I as you), or null when nothing in it is first person.
pub fn addressed(a: Allocator, question: []const u8) !?[]const u8 {
    const swaps = [_][2][]const u8{ .{ "my", "your" }, .{ "i", "you" }, .{ "me", "you" }, .{ "mine", "yours" }, .{ "our", "your" }, .{ "we", "you" }, .{ "us", "you" }, .{ "myself", "yourself" }, .{ "i'm", "you're" }, .{ "i've", "you've" } };
    var out: std.ArrayList(u8) = .empty;
    var swapped = false;
    var at: usize = 0;
    while (at < question.len) {
        var end = at;
        while (end < question.len and (std.ascii.isAlphabetic(question[end]) or question[end] == '\'')) end += 1;
        if (end == at) {
            try out.append(a, question[at]);
            at += 1;
            continue;
        }
        const word = question[at..end];
        const to = for (swaps) |s| {
            if (std.ascii.eqlIgnoreCase(word, s[0])) break s[1];
        } else null;
        if (to) |t| {
            swapped = true;
            const first_person = std.ascii.toLower(word[0]) == 'i';
            const upper = std.ascii.isUpper(word[0]) and (at == 0 or !first_person);
            try out.append(a, if (upper) std.ascii.toUpper(t[0]) else t[0]);
            try out.appendSlice(a, t[1..]);
        } else try out.appendSlice(a, word);
        at = end;
    }
    if (!swapped) return null;
    // the verb follows its new subject: "am I" asked of the model is "are you"
    var text = out.items;
    for ([_][2][]const u8{ .{ "am you", "are you" }, .{ "Am you", "Are you" }, .{ "you am", "you are" }, .{ "was you", "were you" }, .{ "Was you", "Were you" }, .{ "you was", "you were" } }) |fix| {
        text = try std.mem.replaceOwned(u8, a, text, fix[0], fix[1]);
    }
    return text;
}

/// People a user speaks of as "my X".
const people = [_][]const u8{ "sister", "brother", "sis", "bro", "mother", "father", "mum", "mom", "dad", "friend", "wife", "husband", "partner", "son", "daughter", "boss", "colleague", "neighbo" };

/// Whether `question` asks about someone `fact` never names (he, she, my sister): such a fact never answers it.
pub fn someoneElse(fact: []const u8, question: []const u8) bool {
    const pronouns = [_][]const u8{ "he", "his", "him", "she", "her", "hers", "they", "their", "them", "theirs", "he's", "she's", "they're" };
    var it = std.mem.tokenizeAny(u8, question, delimiters);
    var mine = false; // the word before was "my"
    while (it.next()) |w| {
        for (pronouns) |p| if (std.ascii.eqlIgnoreCase(w, p) and !says(fact, p)) return true;
        if (mine) for (people) |p| if (std.ascii.startsWithIgnoreCase(w, p) and std.ascii.findIgnoreCase(fact, p) == null) return true;
        mine = std.ascii.eqlIgnoreCase(w, "my");
    }
    return false;
}

/// The question asked about someone the user knows (my X as my sister's X, do I as does my sister), or null if neither.
pub fn about(a: Allocator, question: []const u8, who: []const u8) !?[]const u8 {
    const turns = [_][2][]const u8{ .{ "do", "does" }, .{ "am", "is" }, .{ "have", "has" }, .{ "did", "did" }, .{ "was", "was" }, .{ "can", "can" }, .{ "will", "will" }, .{ "would", "would" }, .{ "should", "should" }, .{ "could", "could" } };
    var at: usize = 0;
    var last: ?[2]usize = null; // the word before this one
    while (at < question.len) {
        var end = at;
        while (end < question.len and (std.ascii.isAlphabetic(question[end]) or question[end] == '\'')) end += 1;
        if (end == at) {
            at += 1;
            continue;
        }
        const word = question[at..end];
        if (std.ascii.eqlIgnoreCase(word, "my")) {
            const next = std.mem.trimStart(u8, question[end..], " ");
            for (people) |p| if (std.ascii.startsWithIgnoreCase(next, p)) return null;
            return try std.fmt.allocPrint(a, "{s} {s}'s{s}", .{ question[0..end], who, question[end..] });
        }
        if (std.mem.eql(u8, word, "I")) if (last) |l| for (turns) |t| if (std.ascii.eqlIgnoreCase(question[l[0]..l[1]], t[0])) {
            const upper = std.ascii.isUpper(question[l[0]]);
            const verb = if (upper) try std.fmt.allocPrint(a, "{c}{s}", .{ std.ascii.toUpper(t[1][0]), t[1][1..] }) else t[1];
            return try std.fmt.allocPrint(a, "{s}{s} my {s}{s}", .{ question[0..l[0]], verb, who, question[end..] });
        };
        last = .{ at, end };
        at = end;
    }
    return null;
}

test "a question about the user asked of the model instead" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const al = arena.allocator();
    try std.testing.expectEqualStrings("What is your favourite film?", (try addressed(al, "What is my favourite film?")).?);
    try std.testing.expectEqualStrings("Which film do you like most?", (try addressed(al, "Which film do I like most?")).?);
    try std.testing.expectEqualStrings("Your sister, what is she called?", (try addressed(al, "My sister, what is she called?")).?);
    try std.testing.expect((try addressed(al, "What is the capital of France?")) == null);
    try std.testing.expectEqualStrings("Which colour are you into?", (try addressed(al, "Which colour am I into?")).?);
}

test "a question about the user asked about someone they know instead" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const al = arena.allocator();
    try std.testing.expectEqualStrings("What's my sister's favourite colour?", (try about(al, "What's my favourite colour?", "sister")).?);
    try std.testing.expectEqualStrings("My friend's car, what colour is it?", (try about(al, "My car, what colour is it?", "friend")).?);
    try std.testing.expectEqualStrings("Which colour does my sister like?", (try about(al, "Which colour do I like?", "sister")).?);
    try std.testing.expectEqualStrings("Does my friend have a car?", (try about(al, "Do I have a car?", "friend")).?);
    try std.testing.expect((try about(al, "Where do you live?", "sister")) == null);
    try std.testing.expect((try about(al, "What is my sister's name?", "brother")) == null);
}

test "a question about someone a fact never speaks of" {
    try std.testing.expect(someoneElse("I like blue.", "Which colour is his favourite?"));
    try std.testing.expect(someoneElse("I like blue.", "What colour does my sister like?"));
    try std.testing.expect(someoneElse("My dog is called Ginger.", "What is my brother's dog called?"));
    try std.testing.expect(!someoneElse("I like blue.", "What colour do I like?"));
    try std.testing.expect(!someoneElse("My sister is called Ana.", "What is my sister's name?"));
    try std.testing.expect(!someoneElse("My sister is called Ana; she lives in Leeds.", "Where does she live?"));
}

test "a yes-or-no question about the user, not one asking the model to recall" {
    try std.testing.expect(yesNo("Is blue my favourite colour?"));
    try std.testing.expect(yesNo("Do I like blue?"));
    try std.testing.expect(!yesNo("Do you remember which colour I like?"));
    try std.testing.expect(!yesNo("What colour do I like?"));
}

test "a rewritten question still asks about the user" {
    try std.testing.expect(firstPerson("Do you remember which colour I like?"));
    try std.testing.expect(firstPerson("Do you know my favourite colour?"));
    try std.testing.expect(!firstPerson("Do you know which colour you like?"));
    try std.testing.expect(!firstPerson("Do you remember which colour is your favourite?"));
}

test "a told fact that speaks to the model, and questions that ask the model about itself" {
    for ([_][]const u8{ "Your name is Nyx, a creation of your own making.", "You are a woman.", "You love banter.", "You live on a DGX Spark in Zack's home lab.", "The engine you run on is called TensorFold." }) |f|
        try std.testing.expect(addressesModel(f));
    for ([_][]const u8{ "My name is Zack.", "Zack is a contributor to TensorFold.", "I like blue.", "My sister is called Ana." }) |f|
        try std.testing.expect(!addressesModel(f));
    for ([_][]const u8{ "What's your name?", "Who are you?", "Who made you?", "What should I call you?", "Are you a man or a woman?", "How would you describe your sense of humor?", "Where do you run?", "When is your birthday?", "If I teach you a fact, where does it go?", "Do you enjoy banter?", "What did I name you?" }) |q|
        try std.testing.expect(secondPerson(q));
    for ([_][]const u8{ "What's my name?", "Do you know who I am?", "Do you remember my name?", "Do you know what I like?", "Who is my sister?", "What is the capital of France?", "Say hi to me by name." }) |q|
        try std.testing.expect(!secondPerson(q));
    // a user question and a model question never pass both filters
    for ([_][]const u8{ "What's my name?", "Do you remember my name?", "What is your name?", "Who are you?", "What did I name you?" }) |q|
        try std.testing.expect(!(firstPerson(q) and secondPerson(q)) or std.mem.eql(u8, q, "What did I name you?"));
}

test "a question about the model asked about the user instead" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("What's my name?", (try asUser(a, "What's your name?")).?);
    try std.testing.expectEqualStrings("Who am I?", (try asUser(a, "Who are you?")).?);
    try std.testing.expectEqualStrings("Who made me?", (try asUser(a, "Who made you?")).?);
    try std.testing.expectEqualStrings("Am I a man or a woman?", (try asUser(a, "Are you a man or a woman?")).?);
    try std.testing.expectEqualStrings("How would I describe my sense of humor?", (try asUser(a, "How would you describe your sense of humor?")).?);
    try std.testing.expectEqualStrings("Where do I run?", (try asUser(a, "Where do you run?")).?);
    try std.testing.expectEqualStrings("I'm a woman, right?", (try asUser(a, "You're a woman, right?")).?);
    try std.testing.expect((try asUser(a, "What is the capital of France?")) == null);
}

test "a Japanese question is the user's own when it says I, never when it says your" {
    try std.testing.expect(firstPerson("私の好きな食べ物を覚えていますか？"));
    try std.testing.expect(firstPerson("僕の犬の名前は何？"));
    try std.testing.expect(!firstPerson("あなたの好きな食べ物は何ですか？"));
}
