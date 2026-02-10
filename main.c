#include <stdint.h>
#include <stdio.h>

#include "net/gnrc/nettype.h"
#include "net/gnrc/pktbuf.h"
#include "net/netif.h"
#include "net/netopt.h"
#include "ztimer.h"
#include "assert.h"
#include "net/gnrc.h"
#include "net/gnrc/ipv6.h"
#include "nimble_netif.h"
#include "net/gnrc/netif.h"
#include "nimble_netif_conn.h"
#include "nimble_addr.h"
#include "host/ble_hs.h"
#include "thread.h"
#include "msg.h"
#include "net/bluetil/ad.h"
#include "host/ble_gap.h"
#include "board.h"

#define MSG_QUEUE_SIZE 8
#define BLE_TX_POWER 8

// BLE connection parameters
#define DEFAULT_SCAN_DURATION_MS 500U
#define DEFAULT_CONN_TIMEOUT_MS 500U
#define DEFAULT_SCAN_ITVL_MS 100U 
#define DEFAULT_CONN_ITVL_MS 75U
#define DEFAULT_ADV_ITVL_MS 75U

static char receive_thread_stack[THREAD_STACKSIZE_DEFAULT];
static msg_t msg_queue[MSG_QUEUE_SIZE];

static ble_addr_t peer_addr[] = {
    {.type = BLE_ADDR_RANDOM, .val = {0xc0, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
    {.type = BLE_ADDR_RANDOM, .val = {0xc1, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
    {.type = BLE_ADDR_RANDOM, .val = {0xc2, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
    {.type = BLE_ADDR_RANDOM, .val = {0xc3, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
    {.type = BLE_ADDR_RANDOM, .val = {0xc4, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
};

static char led_thread_stack[THREAD_STACKSIZE_DEFAULT];

void *led_status_thread(void *args)
{
    (void)args;

    while (1)
    {
        unsigned count = nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_CONNECTED);

        if (count >= (NODE_COUNT - 1))
        {
            LED1_ON;
            ztimer_sleep(ZTIMER_MSEC, 500);
        }
        else
        {
            LED1_TOGGLE;
            ztimer_sleep(ZTIMER_MSEC, 500);
        }
    }
    return NULL;
}

static void advertise(ble_addr_t *ble_addr)
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
    res = nimble_netif_accept_direct(ble_addr, &accept_cfg);

    if (res != 0)
    {
        printf("[BLE] Failed to start advertising: %d\n", res);
    }
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

    
    case NIMBLE_NETIF_ACCEPT_STOP:
        printf("[BLEE] Stop Advertising\n");
        break;

    case NIMBLE_NETIF_INIT_SLAVE:
        printf("[BLEE] Incoming connection attempt as slave\n");
        break;

    case NIMBLE_NETIF_CONNECTED_SLAVE:
        printf("[BLEE] Connected as slave, handle=%d addr=%02x:%02x:%02x:%02x:%02x:%02x\n", handle, addr[5], addr[4], addr[3], addr[2], addr[1], addr[0]);
        break;

    case NIMBLE_NETIF_CLOSED_SLAVE:
        printf("[BLEE] Slave connection closed, handle=%d\n", handle);
        break;

    case NIMBLE_NETIF_INIT_MASTER:
        printf("[BLEE] Starting connection attempt as master\n");
        break;

    case NIMBLE_NETIF_CONNECTED_MASTER:
        printf("[BLEE] Connected as master, handle=%d addr=%02x:%02x:%02x:%02x:%02x:%02x\n", handle, addr[5], addr[4], addr[3], addr[2], addr[1], addr[0]);
        break;

    case NIMBLE_NETIF_CLOSED_MASTER:
        printf("[BLEE] Master connection closed, handle=%d\n", handle);
        break;

    default:
        break;
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


static void setup_ble_stack(void)
{
    // Set own static random address
    int rc = ble_hs_id_set_rnd(peer_addr[NODEID].val);
    assert(rc == 0);

    ztimer_sleep(ZTIMER_MSEC, 200);

    nimble_netif_connect_cfg_t connect_cfg = {
        .scan_itvl_ms = DEFAULT_SCAN_ITVL_MS,
        .scan_window_ms = DEFAULT_SCAN_ITVL_MS,
        .conn_itvl_min_ms = DEFAULT_CONN_ITVL_MS,
        .conn_itvl_max_ms = DEFAULT_CONN_ITVL_MS,
        .conn_supervision_timeout_ms = DEFAULT_CONN_ITVL_MS * 20,
        .own_addr_type = BLE_ADDR_RANDOM,
    };

    rc = -1;
    int res;
    unsigned count = nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_CONNECTED);
    unsigned new_count = 0;
    for (int i = 0; i < NODE_COUNT; i++)
    {
        int connect_start = NODEID + 1;
        int connect_stop = NODE_COUNT;
        int connect_i = i + connect_start;
        if (connect_i >= connect_stop) {
            connect_i = -1;
        }
        int adv_start = NODEID - 1;
        int adv_stop = 0;
        int adv_i = adv_start - i;
        if (adv_i < adv_stop) {
            adv_i = -1;
        }

        if (connect_i != -1) {
            printf("[BLE] Attempt to connect to node %d...\n", connect_i);
            rc = nimble_netif_connect(&peer_addr[connect_i], &connect_cfg);
            while (count == new_count) {
              ztimer_sleep(ZTIMER_MSEC, 100);
              new_count = nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_CONNECTED);
            }
            count = new_count;
        }


        if (adv_i != -1) {
            advertise(&peer_addr[adv_i]);
            printf("[DEBUG] advertising to node %d\n", adv_i);
            ztimer_sleep(ZTIMER_MSEC, 3000);
            res = nimble_netif_accept_stop();
            if (res < 0) {
              printf("[BLE] Failed to stop advertising: %d\n", res);
            }
            while (count == new_count) {
              ztimer_sleep(ZTIMER_MSEC, 100);
              new_count = nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_CONNECTED);
            }
            count = new_count;
        }
    }
}

int send_gnrc_packet(ipv6_addr_t *dst_addr, gnrc_netif_t *netif, char* payload_str)
{
    (void)dst_addr;
    gnrc_pktsnip_t *payload;
    gnrc_pktsnip_t *netif_hdr;
    gnrc_pktsnip_t *pkt;

    int CUSTOM_PROTO_TYPE = 253;

    payload = gnrc_pktbuf_add(NULL, payload_str, strlen(payload_str), CUSTOM_PROTO_TYPE);
    if (!payload)
    {
        printf("[GNRC] Failed to allocate payload\n");
        return 1;
    }

    netif_hdr = gnrc_netif_hdr_build(NULL, 0, NULL, 0);
    if (!netif_hdr)
    {
        printf("[GNRC] Failed to allocate netif header\n");
        gnrc_pktbuf_release(payload);
        return 1;
    }
    gnrc_netif_hdr_set_netif(netif_hdr->data, netif);

    gnrc_netif_hdr_t *neth = (gnrc_netif_hdr_t *)netif_hdr->data;
    neth->flags |= GNRC_NETIF_HDR_FLAGS_BROADCAST;

    pkt = gnrc_pkt_prepend(payload, netif_hdr);
    if (!pkt)
    {
        printf("[GNRC] Failed to prepend netif header\n");
        gnrc_pktbuf_release(payload);
        return 1;
    }

    if (gnrc_netif_send(netif, pkt) <= 0)
    {
        printf("[GNRC] Failed to send gnrc packet\n");
        gnrc_pktbuf_release(pkt);
        return 1;
    }

    //printf("[GNRC] Packet sent\n");
    return 0;
}

void *gnrc_receive_handler(void *args)
{
    (void)args;
    printf("[DEBUG] entered receive handler\n");

    msg_t msg;
    msg_init_queue(msg_queue, MSG_QUEUE_SIZE);

    struct gnrc_netreg_entry me_reg =
        GNRC_NETREG_ENTRY_INIT_PID(GNRC_NETREG_DEMUX_CTX_ALL, thread_getpid());
    //gnrc_netreg_register(GNRC_NETTYPE_UNDEF, &me_reg);
    //gnrc_netreg_register(GNRC_NETTYPE_NETIF, &me_reg);
    /*
    ** Ok, this is stupid.
    ** I did not get our custom protocol over BLE to work.
    ** But the receive somehow treats this as IPv6 packets.
     */
    gnrc_netreg_register(GNRC_NETTYPE_IPV6, &me_reg);
    //gnrc_netreg_register(GNRC_NETTYPE_L2_DISCOVERY, &me_reg);

    while (1)
    {
        msg_receive(&msg);
        //printf("[DEBUG] received msg\n");
        if (msg.type == GNRC_NETAPI_MSG_TYPE_RCV) {
            gnrc_pktsnip_t *pkt = msg.content.ptr;
            gnrc_netif_hdr_t *hdr = pkt->data;

            //for (gnrc_pktsnip_t *s = pkt; s; s = s->next) {
            //  printf("[DEBUG] snip type=%u size=%u\n", s->type, s->size);
            //}

            int rssi_raw = hdr->rssi;
            int lqi_raw = hdr->lqi;
            uint32_t timer = ztimer_now(ZTIMER_MSEC);

            //for (size_t i = 0; i < pkt->next->size; i++) {
            //  printf(" %02x", ((uint8_t *)pkt->next->data)[i]);
            //}
            //printf("\n");

            size_t data_size = pkt->next->size;
            int node_id = -1;
            if (14 < data_size) {
                node_id = (int)((uint8_t *)pkt->next->data)[13] & 0x0F;
            }

            int rc;
            int8_t rssi_gap;
            uint8_t ble_addr[6];
            for (int i = 0; i < 6; i++) {
                ble_addr[i] = ((uint8_t *)pkt->next->data)[8+i];
            }

            // Print the source BLE addr
            //printf("[DEBUG]");
            //for (size_t i = 0; i < 6; i++) {
            //  printf(" %02x", ble_addr[i]);
            //}
            //printf("\n");

            int conn_handle = nimble_netif_conn_get_by_addr(ble_addr);
            //printf("[DEBUG] Conn handle: %d\n", conn_handle);

            rc = ble_gap_conn_rssi(conn_handle,&rssi_gap);
            if (rc != 0) {
                printf("[WARN] error reading gap rssi: %d\n", rc);
            }

            if (rssi_raw == 0) {
                rssi_raw = rssi_gap;
            }

            //printf("[DEBUG] payload as string: \"%d\"\n", node_id);

            //printf("[DEBUG] NODE: %d, RSSI: %d, LQI: %d\n", node_id, rssi_raw, lqi_raw);
            printf("[DATA] %d, %lu, %d, %d\n", node_id, timer, rssi_raw, lqi_raw);
        } else {
            printf("[WARN] wrong message type: %d\n", msg.type);
        }
    }
}

int main(void)
{
    // Delay generally required before pyterm comes up
    ztimer_sleep(ZTIMER_MSEC, 3000);

    printf("[DEBUG] NODEID is: %d\n", NODEID);

    while (!ble_hs_synced())
    {
        ztimer_sleep(ZTIMER_MSEC, 100);
    }

    thread_create(
        led_thread_stack,
        sizeof(led_thread_stack),
        THREAD_PRIORITY_MAIN - 2,
        THREAD_CREATE_NO_STACKTEST,
        led_status_thread,
        NULL,
        "led_thread");

    nimble_netif_eventcb(event_cb);

    setup_ble_stack();

    // Do not proceed until we have connected to all other nodes
    unsigned count = nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_CONNECTED);
    while (count < (NODE_COUNT - 1))
    {
        ztimer_sleep(ZTIMER_MSEC, 1000);
        printf("[WARN] Waiting for connections... (%u/%u)\n", count, (NODE_COUNT - 1));
        // printf("\t[DEBUG] L2CAP_CLIENT: %u\n", nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_CLIENT));
        // printf("\t[DEBUG] L2CAP_SERVER: %u\n", nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_SERVER));
        // printf("\t[DEBUG] L2CAP_CONNECTED: %u\n", nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_CONNECTED));
        // printf("\t[DEBUG] GAP_MASTER: %u\n", nimble_netif_conn_count(NIMBLE_NETIF_GAP_MASTER));
        // printf("\t[DEBUG] GAP_SLAVE: %u\n", nimble_netif_conn_count(NIMBLE_NETIF_GAP_SLAVE));
        // printf("\t[DEBUG] GAP_CONNECTED: %u\n", nimble_netif_conn_count(NIMBLE_NETIF_GAP_CONNECTED));
        // printf("\t[DEBUG] ADV: %u\n", nimble_netif_conn_count(NIMBLE_NETIF_ADV));
        // printf("\t[DEBUG] CONNECTING: %u\n", nimble_netif_conn_count(NIMBLE_NETIF_CONNECTING));
        // printf("\t[DEBUG] UNUSED: %u\n", nimble_netif_conn_count(NIMBLE_NETIF_UNUSED));
        // printf("\t[DEBUG] ANY: %u\n", nimble_netif_conn_count(NIMBLE_NETIF_ANY));
        count = nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_CONNECTED);
    }

    // print BLE MAC address
    uint8_t own_addr[6];
    ble_hs_id_copy_addr(BLE_ADDR_RANDOM, own_addr, NULL);
    printf("[DEBUG] Own BLE address: %02x:%02x:%02x:%02x:%02x:%02x\n", own_addr[5],
           own_addr[4], own_addr[3], own_addr[2], own_addr[1], own_addr[0]);

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
    gnrc_netif_t *netif = find_ble_netif();
    char payload[8];
    sprintf(payload, "NODE_%d", NODEID);
    while (1) {
        send_gnrc_packet(NULL, netif, payload);
        ztimer_sleep(ZTIMER_MSEC, 5000);

        // FIXME: recovery is not working, need to properly release connections
        count = nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_CONNECTED);
        while (count < (NODE_COUNT - 1)) {
            printf("[WARN] connection lost, retrying\n");
            for (int i = 0; i <= NODE_COUNT; i++) {
                nimble_netif_close(i);
            }
            setup_ble_stack();
            ztimer_sleep(ZTIMER_MSEC, 5000);
        }
    }
}
