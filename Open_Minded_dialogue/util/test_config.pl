%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 		SETTINGS		   %%%
%%%  		Open-Minded Robots		   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SESSION SETTINGS		    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% User settings 		    %%%
userId('000'). % Participant id 
localVariable(first_name_cri, "Unika"). % First name at session start
localVariable(first_name_tablet, "Eunike"). % Child name displayed on tablet, set at session start
localVariable(operator_name, "Julianna"). % Researcher name, set at session start

%%% Session settings		    %%%
multiSessionDesignId(open_minded_memory). % Experimental: open_minded_memory | Control: open_minded_control
sessionId(1). % Values: 1 (single session design)

%%% Condition settings		    %%%
% experimental = C2
% - memory access via tablet memory book
% - tutorial explicitly references the tablet
% - child can inspect Leos memory visually
%
% control = C1
% - memory access via spoken dialogue only
% - tutorial explains verbal memory access
% - no tablet memory interface shown

condition(experimental). % experimental or control

%%% Memory settings		    %%%
% In case of CRI dialogue crash halfway interaction:
% - Stop dialogue (Ctrl+C)
% - Set continueSession to true
% - Restart dialogue 
% - System will ask whether to resume from the previous JSON log  
% - Set continueSession to false before next participant.
continueSession(false). % true or false


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DEFAULT VALUES		    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ROBOT defaults		    %%%
localVariable(robot_name, "Leo").
pauseButton('MiddleTactilTouched').
basePosture('Stand').
tabletUse(experimental). % experimental = show memory book tablet | control = no tablet

%%% UM API defaults		    %%%
umApiBase('http://localhost:8000'). % Eunike's FastAPI + GraphDB
umSource(live). % live = Eunike's API | fake = local JSON personas

%%% SPEECH defaults		    %%%
% Speech-to-text (Whisper)
sttTimeout(20). % Seconds Whisper waits for any speech
sttPhraseLimit(18). % Seconds max for a single phrase
reviewTranscripts(true). % Researcher reviews each transcript before continuing

% Text-to-speech (NAO)
keyValue(default, default, speechSpeed, 85). % NAO speech speed (1-100)
keyValue(default, default, speechLanguage, 'Dutch').

%%% LLM defaults		    %%%
% Intent classification (GPT-4o-mini)
intentModel('gpt-4o-mini').
intentConfidenceThreshold(0.7). % Below this it ask child to repeat

% L3 runtime responses (GPT-4o-mini)
l3Model('gpt-4o-mini').
l3MaxTokens(140).

% Wrong-value generation (deliberate mistakes)
wrongValueModel('gpt-4o-mini').

%%% INPUT MODE defaults		    %%%
% How child responses are captured
childInputMode(microphone). % microphone | keyboard
useDesktopMic(true). % true = laptop mic | false = NAO mic
simulationMode(false). % true = LLM plays the child (testing only)

%%% TABLET defaults		    %%%
tabletServerPort(8080).
tabletPollInterval(2.0). % Seconds between UM polls

%%% SCRIPT defaults		    %%%
scriptVersion('CRI-BRANCH-BASIC4.0').

% Script structure:
% Part 1 = Orientation + Exploratory
% Part 2 = School Bridge + Exploratory/Affective
% Part 3 = Aspiration + Memory Inspection + Closing
%
% Operational dialogue phases include:
% greetings, stories, topic discussions,
% seeded mistakes, nudges, memory inspection,
% and closing sequence.

totalScriptPhases(21).

% Dialogue starts from the first executable phase
startPhase(1).

%%% LOGGING defaults		    %%%
conversationLogEnabled(true).
postPhaseTestControls(true). % Researcher can repeat/skip/quit after each phase
waitForPreviewConfirmation(true). % Pause before starting dialogue

%%% MOVECONFIG defaults		    %%%
% LED eye colors during interaction
keyValue(default, default, eyeColorListening, [0, 1, 0]). % Green = Leo is listening
keyValue(default, default, eyeColorDefault, [1, 1, 1]). % White = default state

%%% FILE PATHS			    %%%
% All resolved relative to Open_Minded_dialogue/
envPath('conf/.env').
conversationLogRoot('_local/conversations').
sessionStatePath('_local/session_state.json').

