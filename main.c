#include <stdint.h>
#include <stdio.h>

#include "net/gnrc/nettype.h"
#include "net/netopt.h"
#include "ztimer.h"
#include "assert.h"
#include "net/gnrc.h"
#include "net/gnrc/ipv6.h"
#include "nimble_netif.h"
#include "nimble_netif_conn.h"
#include "nimble_addr.h"
#include "host/ble_hs.h"
#include "thread.h"
#include "msg.h"
#include "net/bluetil/ad.h"
#include "board.h"

#define NODE_COUNT 5
#define MSG_QUEUE_SIZE 8
#define BLE_TX_POWER 8

// BLE connection parameters
#define DEFAULT_SCAN_DURATION_MS 500U
#define DEFAULT_CONN_TIMEOUT_MS 500U
#define DEFAULT_SCAN_ITVL_MS 100U
#define DEFAULT_CONN_ITVL_MS 75U
#define DEFAULT_ADV_ITVL_MS 75U

static const char *addr_node_str[] = {"2001:db8::1", "2001:db8::2", "2001:db8::3", "2001:db8::4", "2001:db8::5"};
static ipv6_addr_t addr_node[NODE_COUNT];
static char receive_thread_stack[THREAD_STACKSIZE_DEFAULT];
static msg_t msg_queue[MSG_QUEUE_SIZE];

static ble_addr_t peer_addr[] = {
    {.type = BLE_ADDR_RANDOM, .val = {0xc0, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
    {.type = BLE_ADDR_RANDOM, .val = {0xc1, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
    {.type = BLE_ADDR_RANDOM, .val = {0xc2, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
    {.type = BLE_ADDR_RANDOM, .val = {0xc3, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
    {.type = BLE_ADDR_RANDOM, .val = {0xc4, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
};

static gnrc_netif_t *ble_netif = NULL;


static char led_thread_stack[THREAD_STACKSIZE_DEFAULT];

static char led_thread_stack[THREAD_STACKSIZE_DEFAULT];

void *led_status_thread(void *args)
{
    (void)args;

    while (1) {
        unsigned count = nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_CONNECTED);

        if (count >= (NODE_COUNT - 1)) {
            LED1_ON;
            ztimer_sleep(ZTIMER_MSEC, 500);
        }
        else {
            LED1_TOGGLE;
            ztimer_sleep(ZTIMER_MSEC, 500);
        }
    }
    return NULL;
}

static void advertise(void)
{
    int res;
    (void)res;

    uint8_t buf[BLE_HS_ADV_MAX_SZ];
    bluetil_ad_t ad;
    nimble_netif_accept_cfg_t accept_cfg = {
        .own_addr_type = BLE_ADDR_RANDOM,
    };
    /* build advertising data */
    res = bluetil_ad_init_with_flags(&ad, buf, BLE_HS_ADV_MAX_SZ,
                                     BLUETIL_AD_FLAGS_DEFAULT);
    assert(res == BLUETIL_AD_OK);
    
    assert(res == BLUETIL_AD_OK);
    
    // define name according to NODEID
    char name_buf[32];
    snprintf(name_buf, sizeof(name_buf), "LinkQuality-Node%d", NODEID);
    const char *name = name_buf;
    res = bluetil_ad_add(&ad, BLE_GAP_AD_NAME, name, strlen(name));
    if (res != BLUETIL_AD_OK)
    {
        puts("err: the given name is too long");
        return;
    }
    /* start listening for incoming connections */
    res = nimble_netif_accept(ad.buf, ad.pos, &accept_cfg);
    assert(res == 0);
}

static void event_cb(int handle, nimble_netif_event_t event,
                     const uint8_t *addr)
{
    (void)addr;
    switch (event)
    {
    case NIMBLE_NETIF_ACCEPTING:
        printf("[BLEE] Advertising\n");
        break;

    case NIMBLE_NETIF_INIT_SLAVE:
        printf("[BLEE] Incoming connection attempt as slave\n");
        break;

    case NIMBLE_NETIF_CONNECTED_SLAVE:
        printf("[BLEE] Connected as slave, handle=%d\n", handle);
        break;

    case NIMBLE_NETIF_CLOSED_SLAVE:
        printf("[BLEE] Slave connection closed, handle=%d\n", handle);
        break;

    case NIMBLE_NETIF_INIT_MASTER:
        printf("[BLEE] Starting connection attempt as master\n");
        break;

    case NIMBLE_NETIF_CONNECTED_MASTER:
        printf("[BLEE] Connected as master, handle=%d\n", handle);
        break;

    case NIMBLE_NETIF_CLOSED_MASTER:
        printf("[BLEE] Master connection closed, handle=%d\n", handle);
        break;

    default:
        break;
    }
}

static void assign_static_ipv6(gnrc_netif_t *netif, const ipv6_addr_t *addr)
{
    uint8_t flags = GNRC_NETIF_IPV6_ADDRS_FLAGS_STATE_VALID;
    int res = gnrc_netif_ipv6_addr_add(netif, addr, 64, flags);
    if (res < 0)
    {
        printf("Failed to add IPv6 address to interface %u: %d\n",
               netif->pid, res);
    }
    else
    {
        char str[IPV6_ADDR_MAX_STR_LEN];
        printf("Added IPv6 address %s to interface %u\n",
               ipv6_addr_to_str(str, addr, sizeof(str)), netif->pid);
    }
}

static gnrc_netif_t *find_ble_netif(void)
{
    gnrc_netif_t *netif = NULL;
    while ((netif = gnrc_netif_iter(netif)))
    {
        if (netif->device_type == NETDEV_TYPE_BLE)
        {
            return netif;
        }
    }
    return NULL;
}

ipv6_addr_t *get_node_addr(uint8_t node_id)
{
    ipv6_addr_t *rc;
    rc = ipv6_addr_from_str(&addr_node[node_id], addr_node_str[node_id]);
    assert(rc != NULL);
    return &addr_node[node_id];
}

static void setup_ble_stack(void)
{
    // Set own static random address
    int rc = ble_hs_id_set_rnd(peer_addr[NODEID].val);
    
    advertise();
    
    ztimer_sleep(ZTIMER_MSEC, 200);

    nimble_netif_connect_cfg_t connect_cfg = {
        .scan_itvl_ms = DEFAULT_SCAN_ITVL_MS,
        .scan_window_ms = DEFAULT_SCAN_ITVL_MS,
        .conn_itvl_min_ms = DEFAULT_CONN_ITVL_MS,
        .conn_itvl_max_ms = DEFAULT_CONN_ITVL_MS,
        .conn_supervision_timeout_ms = DEFAULT_CONN_ITVL_MS * 20,
        .own_addr_type = BLE_ADDR_RANDOM,
    };

    for (int target = NODEID + 1; target < NODE_COUNT; target++)
    {
        printf("[BLE] Attempt to connect to node %d...\n", target);

        rc = nimble_netif_connect(&peer_addr[target], &connect_cfg);

        if (rc < 0)
        {
            printf("[BLE] Failed to initiate connection to node %d: %d\n", target, rc);
        }
        else
        {
            printf("[BLE] Connected to %d\n", target);
        }

        ztimer_sleep(ZTIMER_MSEC, 500);
    }
}

int send_gnrc_packet(ipv6_addr_t *dst_addr, gnrc_netif_t *netif)
{
    const char *pld = "Payload";
    gnrc_pktsnip_t *payload;
    gnrc_pktsnip_t *ip;
    gnrc_pktsnip_t *netif_hdr;
    gnrc_pktsnip_t *pkt;

    ipv6_addr_t *src_addr = &addr_node[NODEID];
    
    payload = gnrc_pktbuf_add(NULL, pld, strlen(pld), GNRC_NETTYPE_UNDEF);
    if (payload == NULL) {
        printf("[IP] Failed to allocate payload\n");
        return 1;
    }

    ip = gnrc_ipv6_hdr_build(payload, src_addr, dst_addr);
    if (ip == NULL) {
        printf("[IP] Failed to allocate IPv6 header\n");
        gnrc_pktbuf_release(payload);
        return 1;
    }

    netif_hdr = gnrc_netif_hdr_build(NULL, 0, NULL, 0);
    if (netif_hdr == NULL) {
        printf("[IP] Failed to allocate netif header\n");
        gnrc_pktbuf_release(ip);
        return 1;
    }

    gnrc_netif_hdr_set_netif(netif_hdr->data, netif);

    gnrc_netif_hdr_t *neth = (gnrc_netif_hdr_t *)netif_hdr->data;
    neth->flags |= GNRC_NETIF_HDR_FLAGS_BROADCAST;

    pkt = gnrc_pkt_prepend(ip, netif_hdr);
    if (pkt == NULL) {
        printf("[IP] Failed to prepend netif header\n");
        gnrc_pktbuf_release(ip);
        return 1;
    }

    if (gnrc_netapi_dispatch_send(GNRC_NETTYPE_IPV6, 0, pkt) <= 0) {
        printf("[IP] Failed to dispatch IPv6 packet\n");
        gnrc_pktbuf_release(pkt);
        return 1;
    }

    printf("[IP] Packet sent\n");
    return 0;
}


void *gnrc_receive_handler(void *args)
{
    (void)args;

    msg_t msg;
    msg_init_queue(msg_queue, MSG_QUEUE_SIZE);

    struct gnrc_netreg_entry me_reg =
        GNRC_NETREG_ENTRY_INIT_PID(GNRC_NETREG_DEMUX_CTX_ALL, thread_getpid());
    gnrc_netreg_register(GNRC_NETTYPE_UNDEF, &me_reg);

    while (1)
    {
        msg_receive(&msg);
        if (msg.type == GNRC_NETAPI_MSG_TYPE_RCV)
        {
            printf("RCV: 4\n");
            gnrc_pktsnip_t *pkt = msg.content.ptr;
            if (pkt->next)
            {
                if (pkt->next->next)
                {
                    gnrc_netif_hdr_t *hdr = pkt->next->next->data;
                    int rssi_raw = (int)hdr->rssi;
                    int lqi_raw = (int)hdr->lqi;
                    printf("RSSI: %d, LQI: %d", rssi_raw, lqi_raw);
                }
            }
        }
    }
}

int main(void)
{
    // Delay generally required before pyterm comes up
    ztimer_sleep(ZTIMER_MSEC, 3000);

    printf("NODEID is: %d\n", NODEID);

    while (!ble_hs_synced())
    {
        ztimer_sleep(ZTIMER_MSEC, 100);
    }

    thread_create(
        led_thread_stack,
        sizeof(led_thread_stack),
        THREAD_PRIORITY_MAIN - 2, // Priorité légèrement inférieure
        THREAD_CREATE_NO_STACKTEST,
        led_status_thread,
        NULL,
        "led_thread"
    );

    nimble_netif_eventcb(event_cb);

    setup_ble_stack();

    // print BLE MAC address
    uint8_t own_addr[6];
    ble_hs_id_copy_addr(BLE_ADDR_RANDOM, own_addr, NULL);
    printf("Own BLE address: %02x:%02x:%02x:%02x:%02x:%02x\n", own_addr[5],
           own_addr[4], own_addr[3], own_addr[2], own_addr[1], own_addr[0]);

    printf("Got beyond nimble!\n");

    // TODO: find out whether we can get away with raw GNRC without IPv6
    for (int n = 0; n < NODE_COUNT; n++)
    {
        get_node_addr(n);
    }

    // Assign IPv6 address to own BLE interface
    // Might need to wait for the BLE interface to come up
    gnrc_netif_t *netif = find_ble_netif();
    if (netif != NULL && netif != ble_netif)
    {
        ble_netif = netif;
        const ipv6_addr_t *my_addr = &addr_node[NODEID];
        assign_static_ipv6(ble_netif, my_addr);
    }
    else
    {
        printf("Error: no BLE interface\n");
    }

    // Handle incoming messages in separate thread
    thread_create(
        receive_thread_stack,
        sizeof(receive_thread_stack),
        THREAD_PRIORITY_MAIN - 1,
        THREAD_CREATE_NO_STACKTEST,
        gnrc_receive_handler,
        NULL,
        "receive_thread");

    // Continuously send packets
    for (int n = 0; n < NODE_COUNT; n++)
    {
        if (n != NODEID)
        {
            send_gnrc_packet(&addr_node[n], netif);
        }
    }
}
