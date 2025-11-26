using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class PlatformerAgent2D : Agent
{
    [Header("Movement")]
    public float moveSpeed = 7f;
    public float jumpForce = 12f;

    [Header("Environment")]
    public Transform goal;
    public LayerMask groundLayer;

    private Rigidbody2D rb;
    private BoxCollider2D col;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody2D>();
        col = GetComponent<BoxCollider2D>();
    }

    public override void OnEpisodeBegin()
    {
        // Reset velocity and position
        rb.linearVelocity = Vector2.zero;
        transform.position = new Vector3(0f, 1f, 0f);

        // Randomize goal position
        if (goal != null)
        {
            goal.position = new Vector3(Random.Range(12f, 20f), Random.Range(1f, 3f), 0f);
        }

        // Randomize platforms if generator exists
        PlatformGenerator2D pg = FindObjectOfType<PlatformGenerator2D>();
        if (pg != null)
        {
            pg.RandomizePlatforms();
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Velocity
        sensor.AddObservation(rb.linearVelocity.x);
        sensor.AddObservation(rb.linearVelocity.y);

        // Is grounded
        sensor.AddObservation(IsGrounded() ? 1f : 0f);

        // Relative goal position
        if (goal != null)
        {
            sensor.AddObservation(goal.position.x - transform.position.x);
            sensor.AddObservation(goal.position.y - transform.position.y);
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float move = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float jump = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        rb.linearVelocity = new Vector2(move * moveSpeed, rb.linearVelocity.y);

        if (jump > 0.5f && IsGrounded())
        {
            rb.AddForce(Vector2.up * jumpForce, ForceMode2D.Impulse);
        }

        // Time penalty
        AddReward(-0.001f);

        // Fell off
        if (transform.position.y < -6f)
        {
            AddReward(-1f);
            EndEpisode();
        }

        // Reached goal
        if (goal != null)
        {
            float dist = Vector2.Distance(transform.position, goal.position);
            if (dist < 1.0f)
            {
                AddReward(1f);
                EndEpisode();
            }
        }
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
}
