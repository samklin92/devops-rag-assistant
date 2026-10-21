# ImagePullBackOff Troubleshooting

## Overview

ImagePullBackOff occurs when Kubernetes cannot pull a container image and delays subsequent attempts.

## Symptoms

- Pod status shows ImagePullBackOff.
- Container fails to start.
- Pod events report image pull errors.
- The pod may initially show ErrImagePull.

## Possible Causes

### Incorrect Image Name or Tag

The image repository or tag may be incorrect, or the requested image may not exist.

### Registry Authentication

The image may be private, and valid registry credentials may be missing or incorrectly configured.

### Registry Connectivity

Cluster nodes may be unable to reach the registry because of DNS, network, firewall, or registry availability issues.

### Registry Rate Limits

The registry may reject image pull requests after a rate limit is reached.

## Diagnostic Commands

Inspect pod status and events:

```bash
kubectl get pods -n <namespace>
kubectl describe pod <pod-name> -n <namespace>
kubectl get events -n <namespace> --sort-by=.metadata.creationTimestamp
```

Inspect the image configured for the workload:

```bash
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].image}'
```

Check the workload's image configuration:

```bash
kubectl get deployment <deployment-name> -n <namespace> -o yaml
```

Look for events such as manifest unknown, unauthorized, pull access denied, DNS failures, connection timeouts, or rate-limit errors.

## Troubleshooting Steps

1. **Verify the image name and tag.** Compare the configured image with the repository and tag available in the container registry.
2. **Check registry credentials.** For private images, verify that the required image pull secret exists in the same namespace as the pod.
3. **Inspect the image pull secret configuration.** Confirm that the workload references the correct secret under `imagePullSecrets`.
4. **Check registry connectivity.** Investigate DNS resolution, firewall rules, network policies where applicable, and registry availability.
5. **Check registry rate limits.** Review the registry's response and account limits if pulls are being throttled.
6. **Review pod events again.** After identifying the likely cause, use the latest event messages to confirm the diagnosis.

## Remediation

Apply the fix that matches the observed error:

* **Incorrect image or tag:** Correct the image repository and tag in the workload configuration.
* **Missing or invalid credentials:** Create or update the appropriate image pull secret and reference it from the pod or workload.
* **Registry connectivity failure:** Resolve the relevant DNS, network, firewall, or registry availability problem.
* **Rate limiting:** Follow the registry's documented limits and authentication recommendations.
* **Image unavailable:** Publish the required image or select a valid image version.

For a Deployment, update the image when necessary:

```bash
kubectl set image deployment/<deployment-name> <container-name>=<registry>/<image>:<tag> -n <namespace>
```

Monitor the rollout:

```bash
kubectl rollout status deployment/<deployment-name> -n <namespace>
kubectl get pods -n <namespace> -w
```

Confirm that the pod starts successfully and that the image pull errors no longer appear in its events.

## Warnings

* Do not put registry passwords or access tokens directly in manifests, source code, or Git repositories.
* Do not make private container images public merely to bypass authentication problems.
* Avoid deleting pods repeatedly without diagnosing the underlying image pull failure.
* Verify the target namespace and workload before applying changes.
* Follow your organization's security policies when modifying registry credentials or workload configuration.

## Related Runbooks

* `crashloopbackoff.md` — Troubleshooting containers that start but repeatedly crash.
* `oomkilled.md` — Investigating containers terminated because of memory limits.
* `database-connectivity.md` — Diagnosing application database connection failures.
