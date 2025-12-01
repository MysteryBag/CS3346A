using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.Serialization;

public class PlatformerAgent2D : Agent
{
    [Header("Movement")]
    public float moveSpeed = 7f;
    public float jumpForce = 12f;

    [Header("Environment")]
    [Tooltip("Goal the agent must reach.")]
    public Transform goal;
    [Tooltip("Radius to consider the goal collected.")]
    public float goalPickupRadius = 1.0f;
    public LayerMask groundLayer;

    [Header("Rewards")]
    [Tooltip("Reward for reaching the goal.")]
    public float goalReward = 12.0f;
    [Tooltip("Bonus added per consecutive goal in the same life to push back-and-forth runs.")]
    public float sequentialGoalBonus = 6.0f;
    [Tooltip("Reward scale for making progress toward the goal each step (can be negative if moving away).")]
    public float progressRewardScale = 0.08f;
    [Tooltip("Penalty scale applied when increasing distance from the goal (kept smaller to allow repositioning).")]
    public float progressRegressionPenaltyScale = 0.03f;
    [Tooltip("Reward scale for moving in the direction of the goal (dot product of velocity and goal direction, scaled by time).")]
    public float directionRewardScale = 0.03f;
    [Tooltip("Penalty scale when moving away from the goal (applied when velocity points opposite the goal, scaled by time).")]
    public float directionAwayPenaltyScale = 0.015f;
    [Tooltip("Reward multiplier decays each second until the goal is reached to favor quick crossings.")]
    public float goalTimeDecayRate = 0.03f;
    [FormerlySerializedAs("survivalRewardPerStep")]
    [Tooltip("Reward per second for staying alive and attempting to cross.")]
    public float survivalRewardPerSecond = 0.02f;
    [Tooltip("Reward for landing after being airborne to encourage controlled jumps.")]
    public float landingReward = 0.1f;
    [FormerlySerializedAs("noGoalPenaltyPerStep")]
    [Tooltip("Penalty per second when no goal has been collected yet this episode.")]
    public float noGoalPenaltyPerSecond = 0.02f;
    [Tooltip("Penalty per second to discourage idling while grounded and barely moving horizontally.")]
    public float idlePenaltyPerSecond = 0.02f;
    [Tooltip("Horizontal speed threshold under which the idle penalty applies when grounded.")]
    public float idleSpeedThreshold = 0.25f;
    [Tooltip("Penalty per second of episode time to reward faster completions.")]
    public float timePenaltyPerSecond = 0.03f;

    [Header("Visuals")]
    [Tooltip("Renderer used to color the agent. If empty, the first SpriteRenderer on this object will be used.")]
    public SpriteRenderer agentRenderer;
    [Tooltip("Renderer used to color the goal. If empty, the first SpriteRenderer on the goal will be used.")]
    public SpriteRenderer goalRenderer;
    [Tooltip("Color applied to the goal when collected.")]
    public Color goalCollectedColor = Color.yellow;

    [Header("Spawn")]
    [Tooltip("If true, alternate starting side each episode; if false, choose a random side. Ignored when Testing Mode is on.")]
    public bool alternateStartSides = false;
    [Tooltip("If true, always spawn agent/goal at their original placed positions (no jitter/alternation).")]
    public bool testingMode = false;
    [Tooltip("Random offset applied around the agent's initial position each episode.")]
    public Vector2 spawnJitter = new Vector2(0.5f, 0.1f);

    [Header("Goal Placement")]
    [Tooltip("Random offset applied around the platform anchor when placing the goal.")]
    public Vector2 goalPositionJitter = new Vector2(0.75f, 0.5f);

    [Header("Fail Conditions")]
    [Tooltip("Agent dies if it falls below its spawn height by this amount (negative value).")]
    public float fallOffsetFromSpawn = -2.0f;
    [Tooltip("Maximum time (seconds) allowed per episode before timeout.")]
    public float maxEpisodeTime = 30f;
    [Tooltip("Penalty applied if the agent times out.")]
    public float timeoutPenalty = -2f;

    [Header("Perception")]
    [Tooltip("Ray length for obstacle sensing in four cardinal directions.")]
    public float rayLength = 6f;
    [Tooltip("Layers considered as obstacles for rays.")]
    public LayerMask obstacleLayers;

    private Rigidbody2D rb;
    private BoxCollider2D col;
    private Vector3 initialPosition;
    private float deathY = 0f;
    private int goalsCollectedThisEpisode = 0;
    private float lastGoalDistance = 0f;
    private Color activeColor;
    private bool wasGrounded = true;
    private float episodeTimer = 0f;
    private bool useInitialGoalPositionNext = true;
    private Vector3 initialGoalPosition;
    private bool startOnGoalSideNext = false;
    private int cachedObservationSize = 14; // kept for reference when adjusting BehaviorParameters

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody2D>();
        col = GetComponent<BoxCollider2D>();
        initialPosition = transform.position;
        if (agentRenderer == null)
        {
            agentRenderer = GetComponentInChildren<SpriteRenderer>();
        }
        if (goal != null && goalRenderer == null)
        {
            goalRenderer = goal.GetComponentInChildren<SpriteRenderer>();
        }
        if (goal != null)
        {
            initialGoalPosition = goal.position;
        }
        else
        {
            initialGoalPosition = initialPosition;
        }
    }

    public override void OnEpisodeBegin()
    {
        rb.linearVelocity = Vector2.zero;
        episodeTimer = 0f;
        goalsCollectedThisEpisode = 0;
        activeColor = goalCollectedColor;
        ApplyColors(activeColor);

        bool startOnGoalSide;
        if (testingMode)
        {
            startOnGoalSide = false; // always use placed positions
        }
        else if (alternateStartSides)
        {
            startOnGoalSide = startOnGoalSideNext;
            startOnGoalSideNext = !startOnGoalSideNext;
        }
        else
        {
            startOnGoalSide = Random.value > 0.5f;
        }

        PlaceAgentAndGoalOpposite(startOnGoalSide);
        wasGrounded = IsGrounded();
        if (goal != null && goal.gameObject.activeSelf)
        {
            lastGoalDistance = Vector2.Distance(transform.position, goal.position);
        }
        else
        {
            lastGoalDistance = 0f;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Velocity
        sensor.AddObservation(rb.linearVelocity.x);
        sensor.AddObservation(rb.linearVelocity.y);

        // Is grounded
        sensor.AddObservation(IsGrounded() ? 1f : 0f);

        // Obstacle sensing (rays in 8 directions for local awareness)
        sensor.AddObservation(CastDistance(Vector2.right));
        sensor.AddObservation(CastDistance(Vector2.left));
        sensor.AddObservation(CastDistance(Vector2.down));
        sensor.AddObservation(CastDistance(Vector2.up));
        sensor.AddObservation(CastDistance((Vector2.right + Vector2.up).normalized));   // up-right
        sensor.AddObservation(CastDistance((Vector2.right + Vector2.down).normalized)); // down-right
        sensor.AddObservation(CastDistance((Vector2.left + Vector2.up).normalized));    // up-left
        sensor.AddObservation(CastDistance((Vector2.left + Vector2.down).normalized));  // down-left

        // Goal relative position
        if (goal != null && goal.gameObject.activeSelf)
        {
            sensor.AddObservation(goal.position.x - transform.position.x);
            sensor.AddObservation(goal.position.y - transform.position.y);
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
        sensor.AddObservation(goal != null && goal.gameObject.activeSelf ? 1f : 0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float move = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float jump = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        bool groundedNow = IsGrounded();
        episodeTimer += Time.deltaTime;

        // Timeout handling
        if (episodeTimer >= maxEpisodeTime)
        {
            AddReward(timeoutPenalty);
            EndEpisode();
            return;
        }

        rb.linearVelocity = new Vector2(move * moveSpeed, rb.linearVelocity.y);

        if (jump > 0.5f && groundedNow)
        {
            rb.AddForce(Vector2.up * jumpForce, ForceMode2D.Impulse);
        }

        // Shaping: reward progress toward goal (penalize moving away)
        if (goal != null && goal.gameObject.activeSelf)
        {
            float dist = Vector2.Distance(transform.position, goal.position);
            float progress = lastGoalDistance - dist;
            if (progress > 0f)
            {
                AddReward(progress * progressRewardScale);
            }
            else if (progress < 0f)
            {
                AddReward(progress * progressRegressionPenaltyScale);
            }
            lastGoalDistance = dist;

            // Directional reward/penalty based on velocity alignment with goal
            Vector2 toGoal = (goal.position - transform.position);
            Vector2 toGoalDir = toGoal.normalized;
            float dot = Vector2.Dot(rb.linearVelocity, toGoalDir);
            if (dot > 0f)
            {
                AddReward(dot * directionRewardScale * Time.deltaTime);
            }
            else if (dot < 0f)
            {
                AddReward(dot * directionAwayPenaltyScale * Time.deltaTime); // negative when moving away
            }
        }

        // Time and survival shaping
        AddReward(-timePenaltyPerSecond * Time.deltaTime);
        AddReward(survivalRewardPerSecond * Time.deltaTime);
        // Penalty for not having collected a goal yet this episode
        if (goalsCollectedThisEpisode == 0)
        {
            AddReward(-noGoalPenaltyPerSecond * Time.deltaTime);
        }
        // Discourage idling on the ground
        if (groundedNow && Mathf.Abs(rb.linearVelocity.x) < idleSpeedThreshold)
        {
            AddReward(-idlePenaltyPerSecond * Time.deltaTime);
        }


        // Fell off
        if (transform.position.y < deathY)
        {
            AddReward(-2.0f);
            EndEpisode();
        }


        // Check goal collection
        if (goal != null && goal.gameObject.activeSelf)
        {
            float dist = Vector2.Distance(transform.position, goal.position);
            if (dist <= goalPickupRadius)
            {
                float bonus = goalsCollectedThisEpisode * sequentialGoalBonus;
                float timeDecay = Mathf.Exp(-goalTimeDecayRate * episodeTimer);
                AddReward((goalReward + bonus) * timeDecay);
                goalsCollectedThisEpisode++;
                SetGoalCollectedVisual();
                MoveGoalToAlternateSide();
                lastGoalDistance = Vector2.Distance(transform.position, goal.position);
            }
        }

        // Landing reward
        if (groundedNow && !wasGrounded && transform.position.y > deathY)
        {
            AddReward(landingReward);
        }
        wasGrounded = groundedNow;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var a = actionsOut.ContinuousActions;
        a[0] = Input.GetAxisRaw("Horizontal");
        a[1] = Input.GetKey(KeyCode.Space) ? 1f : 0f;
    }

    private bool IsGrounded()
    {
        Vector2 origin = new Vector2(col.bounds.center.x, col.bounds.min.y);
        float extraHeight = 0.05f;
        RaycastHit2D hit = Physics2D.Raycast(origin, Vector2.down, extraHeight + 0.01f, groundLayer);
        return hit.collider != null;
    }

    private float CastDistance(Vector2 dir)
    {
        RaycastHit2D hit = Physics2D.Raycast(transform.position, dir.normalized, rayLength, obstacleLayers);
        if (hit.collider == null)
        {
            return 1f; // no hit within range
        }
        return hit.distance / rayLength; // normalized 0..1
    }

    private void ResetGoalPosition()
    {
        if (goal == null) return;
        goal.gameObject.SetActive(true);
        if (testingMode)
        {
            goal.position = initialGoalPosition;
        }
        else
        {
            goal.position = GetJitteredPosition(initialGoalPosition, goalPositionJitter);
            useInitialGoalPositionNext = false; // next time, place at agent start
        }
    }

    private void MoveGoalToAlternateSide()
    {
        if (goal == null) return;
        if (testingMode)
        {
            goal.position = initialGoalPosition;
            goal.gameObject.SetActive(true);
            return;
        }

        Vector3 target = useInitialGoalPositionNext ? initialGoalPosition : initialPosition;
        goal.position = GetJitteredPosition(target, goalPositionJitter);
        goal.gameObject.SetActive(true);
        useInitialGoalPositionNext = !useInitialGoalPositionNext;
    }

    private Color PickRandomColor()
    {
        // Use a random but bright color from the full 24-bit range to reduce collisions.
        int r = Random.Range(32, 256);
        int g = Random.Range(32, 256);
        int b = Random.Range(32, 256);
        return new Color(r / 255f, g / 255f, b / 255f);
    }

    private void ApplyColors(Color color)
    {
        if (agentRenderer != null)
        {
            agentRenderer.color = color;
        }
        if (goalRenderer != null)
        {
            goalRenderer.color = color;
        }
    }

    private void SetGoalCollectedVisual()
    {
        if (goalRenderer != null)
        {
            goalRenderer.color = goalCollectedColor;
        }
    }

    private void PlaceAgentAndGoalOpposite(bool startOnGoalSide)
    {
        if (testingMode)
        {
            transform.position = initialPosition;
            deathY = transform.position.y + fallOffsetFromSpawn;
            if (goal != null)
            {
                goal.gameObject.SetActive(true);
                goal.position = initialGoalPosition;
                useInitialGoalPositionNext = false;
            }
            else
            {
                useInitialGoalPositionNext = true;
            }
            return;
        }

        Vector3 agentAnchor = startOnGoalSide ? initialGoalPosition : initialPosition;
        Vector3 goalAnchor = startOnGoalSide ? initialPosition : initialGoalPosition;

        transform.position = GetJitteredPosition(agentAnchor, spawnJitter);
        deathY = transform.position.y + fallOffsetFromSpawn;

        if (goal != null)
        {
            goal.gameObject.SetActive(true);
            goal.position = GetJitteredPosition(goalAnchor, goalPositionJitter);
            // Prepare alternating placement for the next collection.
            useInitialGoalPositionNext = goalAnchor == initialPosition;
        }
        else
        {
            useInitialGoalPositionNext = true;
        }
    }

    private Vector3 GetJitteredPosition(Vector3 anchor, Vector2 jitterRange)
    {
        return new Vector3(
            anchor.x + Random.Range(-jitterRange.x, jitterRange.x),
            anchor.y + Random.Range(-jitterRange.y, jitterRange.y),
            anchor.z);
    }
}
