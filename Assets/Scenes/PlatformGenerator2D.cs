using UnityEngine;

public class PlatformGenerator2D : MonoBehaviour
{
    public Transform platformsParent;
    public GameObject platformPrefab;
    public int platformCount = 6;
    public float spacing = 3.5f;
    public float gapVariance = 1.2f;
    public float yVariance = 0.6f;

    void Start()
    {
        if (platformsParent == null)
            platformsParent = this.transform;

        if (platformPrefab != null && platformsParent.childCount == 0)
            GenerateInitialPlatforms();
    }

    public void GenerateInitialPlatforms()
    {
        for (int i = 0; i < platformCount; i++)
        {
            var p = Instantiate(platformPrefab, platformsParent);
            p.name = "Platform_" + i;
            p.transform.position = new Vector3(i * spacing + Random.Range(-gapVariance, gapVariance), Random.Range(-yVariance, yVariance), 0f);
        }
    }

    public void RandomizePlatforms()
    {
        int i = 0;
        foreach (Transform p in platformsParent)
        {
            p.position = new Vector3(i * spacing + Random.Range(-gapVariance, gapVariance), Random.Range(-yVariance, yVariance), 0f);
            i++;
        }
    }
}
